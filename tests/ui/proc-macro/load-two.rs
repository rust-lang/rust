//@ run-pass

#![allow(path_statements)]
#![allow(dead_code)]
//@ proc-macro: derive-atob.rs
//@ proc-macro: derive-ctod.rs
//@ ignore-backends: gcc

#[macro_use]
extern crate derive_atob;
#[macro_use]
extern crate derive_ctod;

#[derive(Copy, Clone)]
#[derive(AToB)]
struct A;

#[derive(CToD)]
struct C;

fn main() {
    B;
    D;
}
