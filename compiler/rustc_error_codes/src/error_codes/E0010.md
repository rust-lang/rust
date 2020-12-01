The value of statics and constants must be known at compile time, and they live
for the entire lifetime of a program. Creating a boxed value allocates memory on
the heap at runtime, and therefore cannot be done at compile time.

Erroneous code example:

```compile_fail,E0010
#![feature(box_syntax)]

const CON : Box<i32> = box 0;
```
