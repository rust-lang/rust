Variadic parameters have been used on a non-C ABI function.

Erroneous code example:

```compile_fail,E0045
#![feature(unboxed_closures)]

extern "rust-call" {
    fn foo(x: u8, ...); // error!
}
```

Rust only supports variadic parameters for interoperability with C code in its
FFI. As such, variadic parameters can only be used with functions which are
using the C ABI. To fix such code, put them in an extern "C" block:

```
extern "C" {
    fn foo (x: u8, ...);
}
```
