Panics the current thread.

This allows a program to terminate immediately and provide feedback
to the caller of the program. `panic!` should be used when a program reaches
an unrecoverable state.

This macro is the perfect way to assert conditions in example code and in
tests. `panic!` is closely tied with the `unwrap` method of both
[`Option`][ounwrap] and [`Result`][runwrap] enums. Both implementations call
`panic!` when they are set to [`None`] or [`Err`] variants.

When using `panic!()` you can specify a string payload, that is built using
the [`format!`] syntax. That payload is used when injecting the panic into
the calling Rust thread, causing the thread to panic entirely.

The behavior of the default `std` hook, i.e. the code that runs directly
after the panic is invoked, is to print the message payload to
`stderr` along with the file/line/column information of the `panic!()`
call. You can override the panic hook using [`std::panic::set_hook()`].
Inside the hook a panic can be accessed as a `&dyn Any + Send`,
which contains either a `&str` or `String` for regular `panic!()` invocations.
To panic with a value of another other type, [`panic_any`] can be used.

[`Result`] enum is often a better solution for recovering from errors than
using the `panic!` macro. This macro should be used to avoid proceeding using
incorrect values, such as from external sources. Detailed information about
error handling is found in the [book].

See also the macro [`compile_error!`], for raising errors during compilation.

[ounwrap]: Option::unwrap
[runwrap]: Result::unwrap
[`std::panic::set_hook()`]: ../std/panic/fn.set_hook.html
[`panic_any`]: ../std/panic/fn.panic_any.html
[`Box`]: ../std/boxed/struct.Box.html
[`Any`]: crate::any::Any
[`format!`]: ../std/macro.format.html
[book]: ../book/ch09-00-error-handling.html

# Current implementation

If the main thread panics it will terminate all your threads and end your
program with code `101`.

# Examples

```should_panic
# #![allow(unreachable_code)]
panic!();
panic!("this is a terrible mistake!");
panic!("this is a {} {message}", "fancy", message = "message");
std::panic::panic_any(4); // panic with the value of 4 to be collected elsewhere
```
