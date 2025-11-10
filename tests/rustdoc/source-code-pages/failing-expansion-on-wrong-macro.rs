// This code crashed because a `if` followed by a `!` was considered a macro,
// creating an invalid class stack.
// Regression test for <https://github.com/rust-lang/rust/issues/148617>.

//@ compile-flags: -Zunstable-options --generate-macro-expansion

enum Enum {
    Variant,
}

pub fn repro() {
    if !matches!(Enum::Variant, Enum::Variant) {}
}
