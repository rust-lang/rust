#![feature(const_trait_impl)]
#![feature(effects)]

struct S;
trait T {}

impl ~const T for S {}
//~^ ERROR expected a trait, found type

fn main() {}
