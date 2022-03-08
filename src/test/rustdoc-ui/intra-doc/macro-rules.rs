// check-pass
#![allow(rustdoc::private_intra_doc_links)]

macro_rules! foo {
    () => {};
}

/// [foo!]
pub fn baz() {}
