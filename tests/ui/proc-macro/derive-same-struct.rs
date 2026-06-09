//@ run-pass

#![allow(path_statements)]
#![allow(dead_code)]
//@ proc-macro: derive-same-struct.rs
//@ ignore-backends: gcc

#[macro_use]
extern crate derive_same_struct;

#[derive(AToB)]
struct A;

fn main() {
    C;
}
