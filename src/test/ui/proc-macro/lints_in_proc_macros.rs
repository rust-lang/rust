// aux-build:bang_proc_macro2.rs

#![feature(proc_macro_hygiene)]
#![allow(unused_macros)]

extern crate bang_proc_macro2;

use bang_proc_macro2::bang_proc_macro2;

fn main() {
    let foobar = 42;
    bang_proc_macro2!();
    //~^ ERROR cannot find value `foobar2` in this scope
    //~| HELP a local variable with a similar name exists
    //~| SUGGESTION foobar
    println!("{}", x);
}
