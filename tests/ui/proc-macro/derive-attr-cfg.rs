//@ run-pass

#![allow(dead_code)]
//@ proc-macro: derive-attr-cfg.rs

extern crate derive_attr_cfg;
use derive_attr_cfg::Foo;

#[derive(Foo)]
#[foo]
struct S {
    #[cfg(any())]
    x: i32
}

fn main() {
}
