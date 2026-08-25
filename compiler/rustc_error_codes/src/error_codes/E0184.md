The `Copy` trait was implemented on a type with a `Drop` implementation.

Erroneous code example:

```compile_fail,E0184
#[derive(Copy)]
struct Foo; // error!

impl Drop for Foo {
    fn drop(&mut self) {
    }
}
```

Explicitly implementing both `Drop` and `Copy` trait on a type is currently
disallowed. This feature can make some sense in theory, but the current
implementation is incorrect and can lead to memory unsafety (see
[issue #20126][iss20126]), so it has been disabled for now.

[iss20126]: https://github.com/rust-lang/rust/issues/20126
