#![feature(track_caller)] //~ WARN the feature `track_caller` is incomplete

#[track_caller]
extern "C" fn f() {}
//~^^ ERROR rust ABI is required to use `#[track_caller]`

fn main() {}
