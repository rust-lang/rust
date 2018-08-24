// aux-build:derive-same-struct.rs
// ignore-stage1

#[macro_use]
extern crate derive_same_struct;

#[derive(AToB)]
struct A;

fn main() {
    C;
}
