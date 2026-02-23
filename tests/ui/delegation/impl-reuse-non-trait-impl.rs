#![allow(incomplete_features)]
#![feature(fn_delegation)]

struct Trait(usize);

reuse impl Trait { self.0 }
//~^ ERROR only trait impls can be reused

fn main() {}
