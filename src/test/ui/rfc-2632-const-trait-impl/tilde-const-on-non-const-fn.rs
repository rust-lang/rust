#![feature(const_trait_impl)]
fn a<T: ~const From<u8>>() {}
//~^ ERROR: `~const` is not allowed

struct S;

impl S {
    fn b<T: ~const From<u8>>() {}
    //~^ ERROR: `~const` is not allowed
}

fn main() {}
