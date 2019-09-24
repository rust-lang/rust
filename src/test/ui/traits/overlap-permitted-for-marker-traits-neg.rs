// run-pass
#![allow(dead_code)]
#![feature(overlapping_marker_traits)]
#![feature(optin_builtin_traits)]

// Overlapping negative impls for `MyStruct` are permitted:
struct MyStruct;
impl !Send for MyStruct {}
impl !Send for MyStruct {}

fn main() {
}
