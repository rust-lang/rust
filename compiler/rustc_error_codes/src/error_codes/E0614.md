Attempted to dereference a variable which cannot be dereferenced.

Erroneous code example:

```compile_fail,E0614
let y = 0u32;
*y; // error: type `u32` cannot be dereferenced
```

Only types implementing `std::ops::Deref` can be dereferenced (such as `&T`).
Example:

```
let y = 0u32;
let x = &y;
// So here, `x` is a `&u32`, so we can dereference it:
*x; // ok!
```
