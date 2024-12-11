//@ run-pass

#![allow(path_statements)]
#![allow(dead_code)]
//@ proc-macro: derive-same-struct.rs

#[macro_use]
extern crate derive_same_struct;

#[derive(AToB)]
struct A;

fn main() {
    C;
}
