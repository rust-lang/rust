#![feature(naked_functions, track_caller)] //~ WARN the feature `track_caller` is incomplete

#[track_caller]
#[naked]
fn f() {}
//~^^^ ERROR cannot use `#[track_caller]` with `#[naked]`

fn main() {}
