//@ proc-macro: rustfmt_2.rs
#![rustfmt::skip]
//~^ ERROR: `rustfmt` is ambiguous

extern crate rustfmt_2;

macro_rules! x {
    () => { use rustfmt_2 as rustfmt; };
}

x!();

fn main() {}
