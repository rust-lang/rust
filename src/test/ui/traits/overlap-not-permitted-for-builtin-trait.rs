#![allow(dead_code)]
#![feature(optin_builtin_traits)]

// Overlapping negative impls for `MyStruct` are not permitted:
struct MyStruct;
impl !Send for MyStruct {}
impl !Send for MyStruct {}
//~^ ERROR conflicting implementations of trait

fn main() {
}
