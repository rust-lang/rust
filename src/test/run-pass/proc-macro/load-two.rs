#![allow(path_statements)]
#![allow(dead_code)]
// aux-build:derive-atob.rs
// aux-build:derive-ctod.rs

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
