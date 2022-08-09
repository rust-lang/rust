#![crate_type = "lib"]

#![warn(unused)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Void {}

pub struct UnStruct {
    x: u32,
    v: Void
}

pub fn match_struct(x: UnStruct) {
    match x {} //~ non-exhaustive patterns: type `UnStruct` is non-empty
}

pub fn match_inhabited_field(x: UnStruct) {
    match x.x {} //~  non-exhaustive patterns: type `u32` is non-empty
                 //~| unreachable expression
}

pub fn match_uninhabited_field(x: UnStruct) {
    match x.v {} // ok
}
