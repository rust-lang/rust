#![feature(track_caller)]

#[track_caller]
extern "C" fn f() {}
//~^^ ERROR `#[track_caller]` requires Rust ABI

fn main() {}
