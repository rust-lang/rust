// Regression test for <https://github.com/rust-lang/rust/issues/111189>.
// This test ensures that it doesn't crash.

#![deny(warnings)]

/// #[rustfmt::skip]
//~^ ERROR unresolved link to `rustfmt::skip`
/// #[clippy::whatever]
//~^ ERROR unresolved link to `clippy::whatever`
pub fn foo() {}
