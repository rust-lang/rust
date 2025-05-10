#![feature(eii)]

#[eii(foo)]
fn x();

#[foo]
fn y(a: u64) -> u64 {
//~^ ERROR E0805
    a
}

fn main() {}
