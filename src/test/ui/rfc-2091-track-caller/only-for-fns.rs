#![feature(track_caller)] //~ WARN the feature `track_caller` is incomplete

#[track_caller]
struct S;
//~^^ ERROR attribute should be applied to function

fn main() {}
