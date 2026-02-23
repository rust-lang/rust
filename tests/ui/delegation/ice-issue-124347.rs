#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    reuse Trait::foo { &self.0 }
    //~^ ERROR failed to resolve delegation callee
}

reuse foo;
//~^ ERROR failed to resolve delegation callee

fn main() {}
