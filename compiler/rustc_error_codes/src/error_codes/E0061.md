An invalid number of arguments was passed when calling a function.

Erroneous code example:

```compile_fail,E0061
fn f(u: i32) {}

f(); // error!
```

The number of arguments passed to a function must match the number of arguments
specified in the function signature.

For example, a function like:

```
fn f(a: u16, b: &str) {}
```

Must always be called with exactly two arguments, e.g., `f(2, "test")`.

Note that Rust does not have a notion of optional function arguments or
variadic functions (except for its C-FFI).
