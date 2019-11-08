#![feature(track_caller)]

#[track_caller] //~ ERROR Rust ABI is required to use `#[track_caller]`
extern "C" fn f() {}

fn main() {}
