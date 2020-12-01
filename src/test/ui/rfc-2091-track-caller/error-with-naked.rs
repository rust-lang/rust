#![feature(naked_functions)]

#[track_caller] //~ ERROR cannot use `#[track_caller]` with `#[naked]`
#[naked]
fn f() {}

struct S;

impl S {
    #[track_caller] //~ ERROR cannot use `#[track_caller]` with `#[naked]`
    #[naked]
    fn g() {}
}

fn main() {}
