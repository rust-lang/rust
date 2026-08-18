//@ known-bug: #154963
#![feature(extern_types, negative_impls)]

unsafe extern "C" {
    type ExternType;
}

impl !Unpin for ExternType {}

fn main() {}
