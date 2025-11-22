//@ check-pass
#![allow(dead_code)]
#![allow(unconstructable_pub_struct)]

pub trait Foo : Send { }

pub struct MyFoo {
    children: Vec<Box<dyn Foo>>,
}

impl Foo for MyFoo { }

pub fn main() { }
