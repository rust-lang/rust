//@ run-pass
#![allow(unused_imports)]
// This test checks for namespace pollution by private tests.
// Tests used to marked as public causing name conflicts with normal
// functions only in test builds.

//@ compile-flags: --test

mod a {
    pub fn foo() -> bool {
        true
    }
}

mod b {
    #[test]
    fn foo() {
        local_name(); // ensure the local name still works
    }

    #[test]
    fn local_name() {}
}

use a::*;
use b::*;

pub fn conflict() {
    let _: bool = foo();
}
