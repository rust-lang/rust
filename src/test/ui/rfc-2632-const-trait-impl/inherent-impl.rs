// compile-flags: -Z parse-only

#![feature(const_trait_impl)]
#![feature(const_trait_bound_opt_out)]
#![allow(incomplete_features)]
#![allow(bare_trait_objects)]

struct S;
trait T {}

impl const T {}
//~^ ERROR `const` cannot modify an inherent impl

fn main() {}
