#![allow(dead_code)]
#![feature(negative_impls)]

// Overlapping negative impls for `MyStruct` are not permitted:
struct MyStruct;
impl !Send for MyStruct {}
impl !Send for MyStruct {}
//~^ ERROR conflicting implementations of trait

fn main() {}
