The `reason` value is missing in a stability attribute.

Erroneous code example:

```compile_fail,E0543
#![feature(staged_api)]
#![stable(since = "1.0.0", feature = "test")]

#[stable(since = "0.1.0", feature = "_deprecated_fn")]
#[rustc_deprecated(
    since = "1.0.0"
)] // invalid
fn _deprecated_fn() {}
```

To fix this issue, you need to provide the `reason` field. Example:

```
#![feature(staged_api)]
#![stable(since = "1.0.0", feature = "test")]

#[stable(since = "0.1.0", feature = "_deprecated_fn")]
#[rustc_deprecated(
    since = "1.0.0",
    reason = "explanation for deprecation"
)] // ok!
fn _deprecated_fn() {}
```

See the [How Rust is Made and “Nightly Rust”][how-rust-made-nightly] appendix
of the Book and the [Stability attributes][stability-attributes] section of the
Rustc Dev Guide for more details.

[how-rust-made-nightly]: https://doc.rust-lang.org/book/appendix-07-nightly-rust.html
[stability-attributes]: https://rustc-dev-guide.rust-lang.org/stability.html
