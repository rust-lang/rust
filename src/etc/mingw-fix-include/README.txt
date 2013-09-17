The purpose of these headers is to fix issues with mingw v4.0, as described in #9246.

This works by adding this directory to GCC include search path before mingw system headers directories, 
so we can intercept their inclusions and add missing definitions without having to modify files in mingw/include.

Once mingw fixes all 3 issues mentioned in #9246, this directory and all references to it from rust/mk/* may be removed.
