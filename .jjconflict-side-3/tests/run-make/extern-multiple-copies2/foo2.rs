#![crate_type = "rlib"]

#[macro_use]
extern crate foo1;

pub fn foo2(a: foo1::A) {
    foo1::foo1(a);
}
