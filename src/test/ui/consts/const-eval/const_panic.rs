#![feature(const_panic)]

fn main() {}

const Z: () = panic!("cheese");
//~^ ERROR this constant cannot be used

const Y: () = unreachable!();
//~^ ERROR this constant cannot be used

const X: () = unimplemented!();
//~^ ERROR this constant cannot be used
