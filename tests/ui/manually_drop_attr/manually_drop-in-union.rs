//! A `#[manually_drop]` type with a destructor is still not allowed inside a union!
#![feature(manually_drop_attr)]

extern crate core;

#[manually_drop]
struct ManuallyDropHasDrop;

impl Drop for ManuallyDropHasDrop {
    fn drop(&mut self) {}
}

union MyUnion {
    x: ManuallyDropHasDrop,
    //~^ ERROR: unions cannot contain fields that may need dropping
}

fn main() {}
