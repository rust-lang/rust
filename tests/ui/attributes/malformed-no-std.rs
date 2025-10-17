#![feature(no_core)]
// these all still apply no_std and then later error
#![no_std = "foo"]
//~^ ERROR malformed `no_std` attribute input
#![no_std("bar")]
//~^ ERROR malformed `no_std` attribute input
#![no_std(foo = "bar")]
//~^ ERROR malformed `no_std` attribute input
#![no_core = "foo"]
//~^ ERROR malformed `no_core` attribute input
#![no_core("bar")]
//~^ ERROR malformed `no_core` attribute input
#![no_core(foo = "bar")]
//~^ ERROR malformed `no_core` attribute input

#[deny(unused_attributes)]
#[no_std]
//~^ ERROR crate-level attribute should be an inner attribute: add an exclamation mark:
#[no_core]
//~^ ERROR crate-level attribute should be an inner attribute: add an exclamation mark:
// to fix compilation
extern crate core;
extern crate std;

fn main() {}
