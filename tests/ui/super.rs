//@ run-pass

#![allow(dead_code)]

pub mod a {
    pub fn f() {}
    pub mod b {
        fn g() {
            super::f();
        }
    }
}

pub fn main() {
}
