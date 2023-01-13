// check-pass

pub fn foo() {}

/// [`foo`](Self::foo) //~ WARNING unresolved link to `Self::foo`
pub use foo as bar;
