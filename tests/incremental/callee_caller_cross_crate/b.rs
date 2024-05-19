//@ aux-build:a.rs
//@ revisions:rpass1 rpass2
//@ compile-flags:-Z query-dep-graph

#![feature(rustc_attrs)]

extern crate a;

#[rustc_clean(except="typeck", cfg="rpass2")]
pub fn call_function0() {
    a::function0(77);
}

#[rustc_clean(cfg="rpass2")]
pub fn call_function1() {
    a::function1(77);
}

pub fn main() { }
