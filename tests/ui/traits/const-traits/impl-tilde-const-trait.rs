#![feature(const_trait_impl)]

struct S;
trait T {}

impl ~const T for S {}
//~^ ERROR expected a trait, found type

fn main() {}
