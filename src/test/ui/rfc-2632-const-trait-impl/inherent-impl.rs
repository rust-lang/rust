#![feature(const_trait_impl)]
#![feature(const_trait_bound_opt_out)]
#![allow(incomplete_features)]
#![allow(bare_trait_objects)]

struct S;
trait T {}

impl const S {}
//~^ ERROR inherent impls cannot be `const`
//~| ERROR const trait impls are not yet implemented

impl const T {}
//~^ ERROR inherent impls cannot be `const`
//~| ERROR const trait impls are not yet implemented

fn main() {}
