//! Check that the `super` keyword correctly allows accessing items from the parent module.
//! This test verifies basic module visibility and path resolution when using `super`.

//@ run-pass

#![allow(dead_code)]

pub mod a {
    pub fn f() {}
    pub mod b {
        fn g() {
            super::f(); // Accessing `f` from module `a` (parent of `b`)
        }
    }
}

pub fn main() {}
