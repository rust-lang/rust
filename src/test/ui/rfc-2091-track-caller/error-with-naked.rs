#![feature(naked_functions, track_caller)]

#[track_caller]
#[naked]
fn f() {}
//~^^^ ERROR cannot use `#[track_caller]` with `#[naked]`

fn main() {}
