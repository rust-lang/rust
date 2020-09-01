Attempted to cast to/from a pointer with an unknown kind.

Erroneous code example:

```compile_fail,E0641
let b = 0 as *const _; // error
```

Type information must be provided if a pointer type being cast from/into another
type which cannot be inferred:

```
// Creating a pointer from reference: type can be inferred
let a = &(String::from("Hello world!")) as *const _; // ok!

let b = 0 as *const i32; // ok!

let c: *const i32 = 0 as *const _; // ok!
```
