#![deny(dead_code)]

enum Foo {} //~ ERROR enum `Foo` is never used

impl Clone for Foo {
    fn clone(&self) -> Foo { loop {} }
}

pub trait PubTrait {
    fn unused_method(&self) -> Self;
}

impl PubTrait for Foo {
    fn unused_method(&self) -> Foo { loop {} }
}

fn main() {}
