#![feature(track_caller)] //~ WARN the feature `track_caller` is incomplete

#[track_caller] //~ ERROR Rust ABI is required to use `#[track_caller]`
extern "C" fn f() {}

fn main() {}
