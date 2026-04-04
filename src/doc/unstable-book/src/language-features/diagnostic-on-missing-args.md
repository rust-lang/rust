# `diagnostic_on_missing_args`

The tracking issue for this feature is: [#152494]

[#152494]: https://github.com/rust-lang/rust/issues/152494

------------------------

The `diagnostic_on_missing_args` feature adds the
`#[diagnostic::on_missing_args(...)]` attribute for declarative macros.
It lets a macro definition customize the diagnostic that is emitted when an invocation ends before
all required arguments were provided.

This attribute currently applies to declarative macros such as `macro_rules!` and `pub macro`.
It only affects diagnostics for incomplete invocations; other matcher failures continue to use the
usual macro diagnostics.

```rust,compile_fail
#![feature(diagnostic_on_missing_args)]

#[diagnostic::on_missing_args(
    message = "pair! is missing its second argument",
    label = "add the missing value here",
    note = "this macro expects a type and a value, like `pair!(u8, 0)`",
)]
macro_rules! pair {
    ($ty:ty, $value:expr) => {};
}

pair!(u8);
```
