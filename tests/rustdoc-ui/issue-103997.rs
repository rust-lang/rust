// check-pass

#![allow(rustdoc::unused_reexport_documentation)]

pub fn foo() {}

/// [`foo`](Self::foo) //~ WARNING unresolved link to `Self::foo`
pub use foo as bar;
