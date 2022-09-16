# raytracing-slow-sol
A quick experiment on how raytraced primitives look like if they could move faster than light 

Demonstrated below is a sphere and a couple of boxes. The sphere moves periodically along the X axis according to `sin` function. The left box is doing the same. The right box rotates around the Y axis.

# Speed of light is 30 units/s
https://user-images.githubusercontent.com/1927568/190577957-94135299-73e5-4e4e-ba49-4d89160725c9.mp4

# Speed of light is 0.5 units/s
https://user-images.githubusercontent.com/1927568/190577964-6a6b2499-3dfe-4b35-b25c-3e01b62086c3.mp4

This repository doesn't contain build scripts.
Dependencies are:

- CUDA
- SDL2
- spdlog

# Notes

The code is ugly, contains copy pasta of itself, etc. I don't care in this case, since I just wanted to get a quick prototype.
Up/Down buttons change the speed of light.

The approach taken here is to iteratively trace rays which are limited in length (say, 1cm at a time), while also moving objects in space(and time). Shadows are only casted on the ground plane, also accounting for the finite speed of light. This is quite inefficient, but it's easy to code. The alternative approach would be to derive a solution of the ray-OBB & ray-sphere intersection equations accounting for the fact that the ray speed is finite and that objects move in time. The key challenge here is that the equation becomes quite ugly and depends on a particular function of time describing the object movement. E.g. if a sphere is moving according to `sin` function, the equation to solve will no longer be a quadratic one. Instead, it's going to look something like `a*t^2 + b*t + c + d * sin^2(t) + e * sin(t)`, which is a PITA to solve. Another idea would be to define a function describing an object movement as a piece-wise function, which is compatible with animation. In this case the solution will require iterating over all intervals of the function and solving a more or less classical quadratic equation (in the case of ray-sphere intersection) on each such interval.
On RTX 3070 Mobility though the iterative approach works relatively quickly, enough to understand how things look like.
