#include <spdlog/spdlog.h>
#include <SDL.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <vector>

namespace
{
    constexpr auto WINDOW_WIDTH = 1024;
    constexpr auto WINDOW_HEIGHT = 1024;
    constexpr auto FB_WIDTH = WINDOW_WIDTH;
    constexpr auto FB_HEIGHT = WINDOW_HEIGHT;
    constexpr auto HORIZONTAL_FOV = 90.0f;
    const dim3 THREADS{ 16, 16 };
    const dim3 BLOCKS{ FB_WIDTH / THREADS.x, FB_HEIGHT / THREADS.y };
}

namespace math
{
    struct mat4 {
        float m[4][4];

        __device__ mat4() {
            identity();
        }

        __device__ mat4(
            const float m11, const float m12, const float m13, const float m14,
            const float m21, const float m22, const float m23, const float m24,
            const float m31, const float m32, const float m33, const float m34,
            const float m41, const float m42, const float m43, const float m44
        ) {
            m[0][0] = m11; m[1][0] = m12; m[2][0] = m13; m[3][0] = m14;
            m[0][1] = m21; m[1][1] = m22; m[2][1] = m23; m[3][1] = m24;
            m[0][2] = m31; m[1][2] = m32; m[2][2] = m33; m[3][2] = m34;
            m[0][3] = m41; m[1][3] = m42; m[2][3] = m43; m[3][3] = m44;
        }

        __device__ float* operator[] (const size_t idx) {
            return m[idx];
        }

        __device__ float4 operator*(const float4& v) const {
            float4 ret;
            ret.x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w;
            ret.y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w;
            ret.z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2] * v.w;
            ret.w = m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3] * v.w;
            return ret;
        }

        __device__ mat4 inverse() const {
            const auto n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
            const auto n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
            const auto n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
            const auto n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

            const auto t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
            const auto t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
            const auto t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
            const auto t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

            const auto det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
            const auto idet = 1.0f / det;

            mat4 ret;

            ret[0][0] = t11 * idet;
            ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
            ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
            ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

            ret[1][0] = t12 * idet;
            ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
            ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
            ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

            ret[2][0] = t13 * idet;
            ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
            ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
            ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

            ret[3][0] = t14 * idet;
            ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
            ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
            ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

            return ret;
        }

        __device__ void identity() {
            memset(m, 0, sizeof(m));
            for (uint32_t i = 0; i < 4; ++i)
                m[i][i] = 1.0f;
        }

        __device__ void rotateY(float angle) {
            identity();
            m[0][0] = cos(angle); 
            m[2][0] = sin(angle);
            m[0][2] = -sin(angle); 
            m[2][2] = cos(angle);
        }
    };
}

namespace scene
{
    struct Sphere
    {
        float4 cr; // center and radius
        float3 color;
        __device__ float3 center() const { return make_float3(cr.x, cr.y, cr.z); }
    };

    struct Plane
    {
        float4 coeff;
        float3 color;
    };

    struct Box
    {
        float3 min, max;
        float3 color;
    };

    constexpr __constant__ float3 ambient{ 20, 20, 20 };
    constexpr __constant__ float3 specularColor{ 255, 255, 255 };
    constexpr __constant__ float shininess = 5.0;
    constexpr __constant__ float3 lightDirection{ -0.3, -0.70710678, -0.70710678 };
    constexpr __constant__ Sphere spheres[1] = {
        Sphere{ {0, 2, -5, 1}, {136, 8, 8} }
    };
    constexpr __constant__ size_t sphereCount = sizeof(spheres) / sizeof(spheres[0]);
    constexpr __constant__ Box boxes[2] = {
        Box {
            {-0.7,-0.7,-0.7}, {0.7, 0.7, 0.7},
            {120, 80, 200}
        },
        Box {
            {-0.7,-0.7,-0.7}, {0.7, 0.7, 0.7},
            {220, 180, 200}
        }
    };
    constexpr __constant__ size_t boxCount = sizeof(boxes) / sizeof(boxes[0]);
    __constant__ Plane planes[1] = {
        Plane { {0, 1, 0, 7}, {50, 50, 70} }
    };
    constexpr __constant__ size_t planeCount = sizeof(planes) / sizeof(planes[0]);
}

namespace math
{
    __device__ float radians(float degrees)
    {
        return degrees * M_PI / 180.0f;
    }

    inline __device__ float3 operator+(float3 a, float3 b)
    {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    inline __device__ float3 operator-(float3 a)
    {
        return make_float3(-a.x, -a.y, -a.z);
    }

    inline __device__ float3 operator-(float3 a, float3 b)
    {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    inline __device__ float3 operator*(float3 a, float b)
    {
        return make_float3(a.x * b, a.y * b, a.z * b);
    }

    inline __device__ float3 operator*(float3 a, float3 b)
    {
        return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    inline __device__ float4 operator*(float4 a, float4 b)
    {
        return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
    }

    inline __device__ float dot(float3 a, float3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline __device__ float3 normalize(float3 v)
    {
        return v * rsqrtf(dot(v, v));
    }

    inline __device__ float sign(float x)
    {
        return x > 0 ? 1 : (x < 0 ? -1 : 0);
    }

    inline __device__ float3 sign(float3 x)
    {
        return make_float3(sign(x.x), sign(x.y), sign(x.z));
    }

    inline __device__ float3 abs(float3 v)
    {
        return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
    }

    inline __device__ float3 pow(float3 v, float k)
    {
        return make_float3(powf(v.x, k), powf(v.y, k), powf(v.z, k));
    }

    inline __device__ float3 step(float3 a, float3 b)
    {
        return make_float3(a.x >= b.x ? 1.0f : 0.0f, a.y >= b.y ? 1.0f : 0.0f, a.z >= b.z ? 1.0f : 0.0f);
    }

    inline __device__ float3 reflect(float3 in, float3 n)
    {
        return in - n * 2.f * dot(in, n);
    }

    __device__ bool intersect(float3 origin, float3 direction, const scene::Sphere& sp, float& t)
    {
        const float3 L = sp.center() - origin;
        const float tca = dot(L, direction);

        if (tca < 0)
            return false;

        const float s2 = (dot(L, L)) - (tca * tca);
        const float s = sqrt(s2);

        if (s > sp.cr.w)
            return false;

        t = tca - sqrt((sp.cr.w * sp.cr.w) - s2);
        return true;
    }

    __device__ float4 intersect(float3 origin, float3 direction, const scene::Box& b)
    {
        const float3 p = origin;

        if ((origin.x < b.max.x && origin.x > b.min.x) &&
            (origin.y < b.max.y && origin.y > b.min.y) &&
            (origin.z < b.max.z && origin.z > b.min.z))
        {
            const float3 center = (b.max + b.min) * 0.5f;
            const float3 n = origin - center;
            const float3 dim = (b.min - b.max) * 0.5f;
            const float bias = 1.01f;
            float3 normal{float(int(n.x / fabsf(dim.x) * bias)), 
            float(int(n.y / fabsf(dim.y) * bias)), 
            float(int(n.z / fabsf(dim.z) * bias))};
            normal = normalize(normal);
            return {0.01f, normal.x, normal.y, normal.z};
        }
        return {-1.0f};
    }

    __device__ float intersect(float3 origin, float3 direction, const scene::Plane& b)
    {
        return -(dot(origin, make_float3(b.coeff.x, b.coeff.y, b.coeff.z)) + b.coeff.w) / dot(direction, make_float3(b.coeff.x, b.coeff.y, b.coeff.z));
    }
}

__device__ void setColor(uint8_t* displayGpu, uint16_t x, uint16_t y, uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t* base = displayGpu + 3 * (y * FB_WIDTH + x);
    base[0] = r;
    base[1] = g;
    base[2] = b;
}

__global__ void raytraceKernel(uint8_t* displayGpu, float globalTime, float C)
{
    using namespace scene;
    using namespace math;

    const auto phongBRDF = [](float3 lightDir, float3 viewDir, float3 normal, float3 phongDiffuseCol, float3 phongSpecularCol, float phongShininess) {
        float3 color = phongDiffuseCol;
        float3 reflectDir = reflect(-lightDir, normal);
        float specDot = fmaxf(dot(reflectDir, viewDir), 0.0);
        color = color + phongSpecularCol * pow(specDot, phongShininess);
        return color;
    };

    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    const float pixNormX = (x + 0.5f) / FB_WIDTH;
    const float pixNormY = (y + 0.5f) / FB_HEIGHT;
    const float imageAspectRatio = FB_WIDTH / FB_HEIGHT;
    const float pixRemapX = (2 * pixNormX - 1) * imageAspectRatio;
    const float pixRemapY = 1 - 2 * pixNormY;
    const float pixCameraX = pixRemapX * tan(radians(HORIZONTAL_FOV) / 2);
    const float pixCameraY = pixRemapY * tan(radians(HORIZONTAL_FOV) / 2);

    const float3 camera{ pixCameraX, pixCameraY, -1 };
    const float3 origin{0,0,0};
    float3 dir = normalize(camera - origin);

    const uint32_t iterations = 1280;
    const float maxDistance = 7.0f;
    const float distancePerIteration = maxDistance / iterations;
    const float timePerIteration = distancePerIteration / C;
    for (uint32_t k = 512; k < iterations; ++k)
    {
        const auto localTime = k * timePerIteration;
        const auto time = globalTime - localTime;
        float3 o = origin + dir * distancePerIteration * k;

        for (size_t i = 0; i < sphereCount; ++i)
        {
            const auto scr = spheres[i].cr;

            const Sphere s{ {scr.x + sinf(time), scr.y, scr.z, scr.w}, spheres[i].color };
            float t;
            if (intersect(o, dir, s, t))
            {
                if (t <= distancePerIteration)
                {
                    const float3 hit = origin + dir * t + dir * (k - 1) * distancePerIteration;
                    const float3 normal = normalize(hit - s.center());
                    float3 color = s.color * fmaxf(0.0f, dot(normal, -lightDirection));
                    const float3 R = reflect(-lightDirection, normal);
                    const float specAngle = fmaxf(dot(R, dir), 0.0);
                    const float shininess = 5.0f;
                    const float specular = pow(specAngle, shininess);
                    color = color + float3{ 255, 255, 255 } * specular * 0.4;
                    setColor(displayGpu, x, y, color.x, color.y, color.z);
                    return;
                }
            }
        }

        for (size_t i = 0; i < boxCount; ++i)
        {
            Box b = boxes[i];
            mat4 identity;

            if (i == 0)
            {
                identity.m[3][0] = -2 + sinf(time / 2);
                identity.m[3][1] = -2;
                identity.m[3][2] = -5;
            }
            else
            {
                identity.rotateY(time / 3);
                identity.m[3][0] = 2;
                identity.m[3][1] = -2;
                identity.m[3][2] = -5;
            }
            const mat4 inv = identity.inverse();
            const float4 o4 = make_float4(o.x, o.y, o.z, 1);
            const float4 d4 = make_float4(dir.x, dir.y, dir.z, 0);
            const float4 invO4 = inv * o4;
            const float4 invDir4 = inv * d4;
            const float3 c = (b.min + b.max) * 0.5f;
            const auto oB = float3{ invO4.x, invO4.y, invO4.z };
            const auto dirB = normalize(float3{ invDir4.x, invDir4.y, invDir4.z });

            const float4 maybeHit = intersect(oB, dirB, b);
            if (maybeHit.x > 0)
            {
                float3 baseColor = boxes[i].color;
                const float3 c = (b.min + b.max) * 0.5f;
                const float3 bo{ (oB.x - c.x) * 10.0f, (oB.y - c.y) * 10.0f, (oB.z - c.z) * 10.0f };
                
                if ((int32_t(bo.x) + int32_t(bo.y) + int32_t(bo.z)) % 2 == 0)
                    baseColor = baseColor * 0.1;
                const float3 normal{ maybeHit.y, maybeHit.z, maybeHit.w };
                const float3 viewDir = normalize(origin - o);
                const float3 lightDir = normalize(-lightDirection);
                const float irradiance = fmaxf(dot(lightDir, normal), 0.0);
                float3 radiance = baseColor;
                if (irradiance > 0.0) 
                {
                    const float3 brdf = phongBRDF(lightDir, viewDir, normal, radiance, specularColor, shininess);
                    radiance = radiance + brdf * irradiance;
                }
                const float3 color = pow(radiance, 1.0 / 2.2) * 20;

                setColor(displayGpu, x, y, color.x, color.y, color.z);
                return;
            }
        }
    }


    for (size_t i = 0; i < planeCount; ++i)
    {
        const auto t = intersect(origin, dir, planes[i]);
        if (t > 0.0f)
        {
            const float3 hit = origin + dir * t;
            float3 baseColor = planes[i].color;
            if ((int32_t(hit.x) + int32_t(hit.z)) % 2 == 0)
                baseColor = baseColor * 0.1;
            const float3 normal{ planes[i].coeff.x, planes[i].coeff.y, planes[i].coeff.z };
            const float3 viewDir = normalize(origin - hit);
            const float3 lightDir = normalize(-lightDirection);
            const float irradiance = fmaxf(dot(lightDir, normal), 0.0);
            float3 radiance = ambient;
            if (irradiance > 0.0) 
            {
                const float3 brdf = phongBRDF(lightDir, viewDir, normal, baseColor, specularColor, shininess);
                radiance = radiance + brdf * irradiance;
            }

            float3 color = pow(radiance, 1.0 / 2.2) * 10;
            const float timeToPlaneHit = t / C;
            for (uint32_t k = 0; k < iterations; ++k)
            {
                const auto localTime = k * timePerIteration;
                const auto time = globalTime - localTime - timeToPlaneHit;
                float3 o = hit - lightDirection * distancePerIteration * k * 3.0f;

                bool hitFound{};
                for (size_t i = 0; i < sphereCount; ++i)
                {
                    const auto scr = spheres[i].cr;
                    const Sphere s{ {scr.x + sinf(time), scr.y, scr.z, scr.w}, {} };

                    float t;
                    if (intersect(o, -lightDirection, s, t))
                    {
                        if (t <= distancePerIteration)
                        {
                            color = color * 0.5;
                            hitFound = true;
                            break;
                        }
                    }
                }
                if (hitFound)
                    break;

                for (size_t i = 0; i < boxCount; ++i)
                {
                    Box b = boxes[i];
                    auto oB = hit - lightDirection * distancePerIteration * k * 3.0f;;
                    auto dirB = -lightDirection;
                    mat4 identity;

                    if (i == 0)
                    {
                        identity.m[3][0] = -2 + sinf(time / 2);
                        identity.m[3][1] = -2;
                        identity.m[3][2] = -5;
                    }
                    else
                    {
                        identity.rotateY(time / 3);
                        identity.m[3][0] = 2;
                        identity.m[3][1] = -2;
                        identity.m[3][2] = -5;
                    }
                    const mat4 inv = identity.inverse();
                    const float4 o4 = make_float4(oB.x, oB.y, oB.z, 1);
                    const float4 d4 = make_float4(dirB.x, dirB.y, dirB.z, 0);
                    const float4 invO4 = inv * o4;
                    const float4 invDir4 = inv * d4;
                    const float3 c = (b.min + b.max) * 0.5f;
                    oB = float3{ invO4.x, invO4.y, invO4.z };
                    dirB = normalize(float3{ invDir4.x, invDir4.y, invDir4.z });

                    const float4 maybeHit = intersect(oB, dirB, b);
                    if (maybeHit.x > 0)
                    {
                        color = color * 0.5;
                        hitFound = true;
                        break;
                    }
                }

                if (hitFound)
                    break;
            }

            setColor(displayGpu, x, y, color.x, color.y, color.z);
            return;
        }
    }

    setColor(displayGpu, x, y, 0, 0, 0);
}

void setup_logging()
{
    spdlog::set_pattern("[%c %z] [%^%L%$] %v");
    spdlog::set_level(spdlog::level::debug);
}

int main(int, char**)
{
    setup_logging();
    bool quit = false;

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
        spdlog::error("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

    SDL_Event event;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("rt-sol", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_RendererInfo info{};
    SDL_GetRendererInfo(renderer, &info);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, FB_WIDTH, FB_HEIGHT);

    std::vector<uint8_t> display;
    uint8_t* displayGpu{};
    display.resize(FB_WIDTH * FB_HEIGHT * 3);

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaStatus = cudaMalloc((void**)&displayGpu, FB_HEIGHT * FB_WIDTH * 3);
    if (cudaStatus != cudaSuccess)
        spdlog::error("cudaMalloc failed!");

    float t{};
    float C{ 0.5f };
    while (!quit)
    {
        const auto t1 = std::chrono::high_resolution_clock::now();
        raytraceKernel << <BLOCKS, THREADS >> > (displayGpu, t, C);
        t += 0.032;

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
            spdlog::error("emulateKernel launch failed: {}", cudaGetErrorString(cudaStatus));

        const auto t2 = std::chrono::high_resolution_clock::now();
        //spdlog::info("CUDA: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

        cudaMemcpy(display.data(), displayGpu, FB_HEIGHT * FB_WIDTH * 3, cudaMemcpyDeviceToHost);

        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        SDL_UpdateTexture(texture, nullptr, display.data(), FB_WIDTH * 3);

        while (SDL_PollEvent(&event))
        {
            SDL_PumpEvents();
            quit = event.type == SDL_QUIT;
        }

        int len{};
        const uint8_t* keys = SDL_GetKeyboardState(&len);
        if (keys[SDL_SCANCODE_UP])
        {
            C += 0.05f;
            spdlog::info("C = {}", C);
        }
        else if (keys[SDL_SCANCODE_DOWN])
        {
            C -= 0.05f;
            spdlog::info("C = {}", C);
        }

        // 60hz
        using namespace std::chrono_literals;
        //std::this_thread::sleep_for(16ms);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    cudaFree(displayGpu);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}