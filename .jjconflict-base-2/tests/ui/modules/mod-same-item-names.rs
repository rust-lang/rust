//! Test that items with identical names can coexist in different modules

//@ run-pass

#![allow(dead_code)]

mod foo {
    pub fn baz() {}
}

mod bar {
    pub fn baz() {}
}

pub fn main() {}
