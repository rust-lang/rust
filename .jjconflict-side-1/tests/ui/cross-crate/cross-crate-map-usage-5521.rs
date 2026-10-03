// https://github.com/rust-lang/rust/issues/5521
//@ run-pass
#![allow(dead_code)]
//@ aux-build:aux-5521.rs

extern crate aux_5521 as foo;

fn bar(a: foo::map) {
    if false {
        panic!();
    } else {
        let _b = &(*a)[&2];
    }
}

fn main() {}
