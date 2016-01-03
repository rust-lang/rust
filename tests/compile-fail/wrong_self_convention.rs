#![feature(plugin)]
#![plugin(clippy)]

#![deny(wrong_self_convention)]
#![deny(wrong_pub_self_convention)]
#![allow(dead_code)]

fn main() {}

#[derive(Clone, Copy)]
struct Foo;

impl Foo {

    fn as_i32(self) {}
    fn into_i32(self) {}
    fn is_i32(self) {}
    fn to_i32(self) {}
    fn from_i32(self) {} //~ERROR: methods called `from_*` usually take no self

    pub fn as_i64(self) {}
    pub fn into_i64(self) {}
    pub fn is_i64(self) {}
    pub fn to_i64(self) {}
    pub fn from_i64(self) {} //~ERROR: methods called `from_*` usually take no self

}

struct Bar;

impl Bar {

    fn as_i32(self) {} //~ERROR: methods called `as_*` usually take self by reference
    fn into_i32(&self) {} //~ERROR: methods called `into_*` usually take self by value
    fn is_i32(self) {} //~ERROR: methods called `is_*` usually take self by reference
    fn to_i32(self) {} //~ERROR: methods called `to_*` usually take self by reference
    fn from_i32(self) {} //~ERROR: methods called `from_*` usually take no self

    pub fn as_i64(self) {} //~ERROR: methods called `as_*` usually take self by reference
    pub fn into_i64(&self) {} //~ERROR: methods called `into_*` usually take self by value
    pub fn is_i64(self) {} //~ERROR: methods called `is_*` usually take self by reference
    pub fn to_i64(self) {} //~ERROR: methods called `to_*` usually take self by reference
    pub fn from_i64(self) {} //~ERROR: methods called `from_*` usually take no self

}
