//@ build-fail
//@ compile-flags:-Zpolymorphize=on
#![feature(rustc_attrs)]

// This test checks that the polymorphization analysis doesn't break when the
// function/closure doesn't just have generic parameters.

// Function has an unused generic parameter.
#[rustc_polymorphize_error]
pub fn unused<'a, T>(_: &'a u32) {
    //~^ ERROR item has unused generic parameters
}

#[rustc_polymorphize_error]
pub fn used<'a, T: Default>(_: &'a u32) -> u32 {
    let _: T = Default::default();
    let add_one = |x: u32| x + 1;
    //~^ ERROR item has unused generic parameters
    add_one(3)
}

fn main() {
    unused::<u32>(&3);
    used::<u32>(&3);
}
