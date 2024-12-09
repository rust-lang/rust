//@ check-pass
// https://github.com/rust-lang/rust/issues/103997

pub fn foo() {}

/// [`foo`](Self::foo) //~ WARNING unresolved link to `Self::foo`
pub use foo as bar;
