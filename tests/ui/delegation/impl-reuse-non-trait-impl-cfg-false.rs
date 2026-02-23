#![allow(incomplete_features)]
#![feature(fn_delegation)]

struct Trait(usize);

#[cfg(false)]
reuse impl Trait { self.0 }
//~^ ERROR only trait impls can be reused

fn main() {}
