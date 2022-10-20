#![feature(const_trait_impl)]
#![allow(bare_trait_objects)]
#![feature(effects)]

struct S;
trait T {}

impl const S {}
//~^ ERROR inherent impls cannot be `const`
//~| ERROR `host` is not constrained

impl const T {}
//~^ ERROR inherent impls cannot be `const`
//~| ERROR `host` is not constrained

fn main() {}
