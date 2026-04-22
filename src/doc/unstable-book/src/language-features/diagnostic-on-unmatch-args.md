# `diagnostic_on_unmatch_args`

The tracking issue for this feature is: [#155642]

[#155642]: https://github.com/rust-lang/rust/issues/155642

------------------------

The `diagnostic_on_unmatch_args` feature adds the
`#[diagnostic::on_unmatch_args(...)]` attribute for declarative macros.
It lets a macro definition customize diagnostics for matcher failures after all arms have been
tried, such as incomplete invocations or trailing extra arguments.

This attribute currently applies to declarative macros such as `macro_rules!` and `pub macro`.
It is currently used for errors emitted by declarative macro matching itself; fragment parser
errors still use their existing diagnostics.

```rust,compile_fail
#![feature(diagnostic_on_unmatch_args)]

#[diagnostic::on_unmatch_args(
    message = "invalid arguments to {This} macro invocation",
    label = "expected a type and value here",
    note = "this macro expects a type and a value, like `pair!(u8, 0)`",
    note = "see <link/to/docs>",
)]
macro_rules! pair {
    ($ty:ty, $value:expr) => {};
}

pair!(u8);
```

This emits output like:

```text
error: invalid arguments to pair macro invocation
  --> example.rs:13:9
   |
9  | macro_rules! pair {
   | ----------------- when calling this macro
...
13 | pair!(u8);
   |         ^ expected a type and value here
   |
note: while trying to match `,`
  --> example.rs:10:12
   |
10 |     ($ty:ty, $value:expr) => {};
   |            ^
   = note: this macro expects a type and a value, like `pair!(u8, 0)`
   = note: see <link/to/docs>
```
