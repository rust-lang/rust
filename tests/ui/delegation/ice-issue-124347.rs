#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    reuse Trait::foo { &self.0 }
    //~^ ERROR recursive delegation is not supported yet
}

reuse foo;
//~^ ERROR recursive delegation is not supported yet

fn main() {}
