#![feature(const_trait_impl)]

struct Bug {
    inner: [(); match || 1 {
        n => n(),
        //~^ ERROR the trait bound
        //~| ERROR cannot call non-const fn `Bug::inner::{constant#0}::{closure#0}` in constants
    }],
}

fn main() {}
