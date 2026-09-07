//@ edition:2024
//@ compile-flags: -Znext-solver

#![allow(dead_code)]

trait MyTrait {}
impl MyTrait for () {}

impl<'de> DeserTrait<'de> for &'de DeserStruct {}

trait DeserTrait<'de> {}
struct DeserStruct;

impl DeserTrait<'_> for &'static MyTrait {}
//~^ ERROR expected a type, found a trait

fn test() -> impl Send {
    testfn(&DeserStruct)
}

fn testfn<'de, D: DeserTrait<'de>>(_deserializer: D) -> impl MyTrait + 'static {}

fn main() {}
