#![feature(const_trait_bound_opt_out)]
#![feature(const_trait_impl)]
#![allow(incomplete_features)]

struct S;
trait T {}

impl ?const T for S {}
//~^ ERROR expected a trait, found type

fn main() {}
