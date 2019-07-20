#![feature(track_caller)]

#[track_caller]
extern "C" fn f() {}
//~^^ ERROR rust ABI is required to use `#[track_caller]`

fn main() {}
