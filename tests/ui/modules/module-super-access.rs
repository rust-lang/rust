//! Check path resolution using `super`

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
