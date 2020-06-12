Panics the current thread.

This allows a program to terminate immediately and provide feedback
to the caller of the program. `panic!` should be used when a program reaches
an unrecoverable state.

This macro is the perfect way to assert conditions in example code and in
tests. `panic!` is closely tied with the `unwrap` method of both [`Option`]
and [`Result`][runwrap] enums. Both implementations call `panic!` when they are set
to None or Err variants.

This macro is used to inject panic into a Rust thread, causing the thread to
panic entirely. Each thread's panic can be reaped as the `Box<Any>` type,
and the single-argument form of the `panic!` macro will be the value which
is transmitted.

[`Result`] enum is often a better solution for recovering from errors than
using the `panic!` macro. This macro should be used to avoid proceeding using
incorrect values, such as from external sources. Detailed information about
error handling is found in the [book].

The multi-argument form of this macro panics with a string and has the
[`format!`] syntax for building a string.

See also the macro [`compile_error!`], for raising errors during compilation.

[runwrap]: ../std/result/enum.Result.html#method.unwrap
[`Option`]: ../std/option/enum.Option.html#method.unwrap
[`Result`]: ../std/result/enum.Result.html
[`format!`]: ../std/macro.format.html
[`compile_error!`]: ../std/macro.compile_error.html
[book]: ../book/ch09-00-error-handling.html

# Current implementation

If the main thread panics it will terminate all your threads and end your
program with code `101`.

# Examples

```should_panic
# #![allow(unreachable_code)]
panic!();
panic!("this is a terrible mistake!");
panic!(4); // panic with the value of 4 to be collected elsewhere
panic!("this is a {} {message}", "fancy", message = "message");
```
