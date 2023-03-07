#![feature(const_trait_impl)]
#![allow(bare_trait_objects)]

struct S;
trait T {}

impl const S {}
//~^ ERROR inherent impls cannot be `const`

impl const T {}
//~^ ERROR inherent impls cannot be `const`

fn main() {}
