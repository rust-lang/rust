#![feature(track_caller)] //~ WARN the feature `track_caller` is incomplete

#[track_caller(1)]
fn f() {}
//~^^ ERROR malformed `track_caller` attribute input

fn main() {}
