//@ check-pass
#![allow(rustdoc::private_intra_doc_links)]

macro_rules! foo {
    () => {};
}

/// [foo!]
pub fn baz() {}

#[macro_use]
mod macros {
    macro_rules! escaping {
        () => {};
    }
}

pub mod inner {
    /// [foo!]
    /// [escaping]
    pub fn baz() {
        foo!();
    }
}
