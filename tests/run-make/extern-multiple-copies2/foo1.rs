#![crate_type = "rlib"]

pub struct A;

pub fn foo1(a: A) {
    drop(a);
}
