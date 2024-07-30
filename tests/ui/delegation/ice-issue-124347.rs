#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    reuse Trait::foo { &self.0 }
    //~^ ERROR recursive delegation is not supported yet
}

// FIXME(fn_delegation): `recursive delegation` error should be emitted here
reuse foo;
//~^ ERROR cycle detected when computing generics of `foo`

fn main() {}
