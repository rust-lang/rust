#![feature(fn_delegation)]
#![allow(incomplete_features)]

// FIXME(fn_delegation): `recursive delegation` error should be emitted here
trait Trait {
    reuse Trait::foo { &self.0 }
    //~^ ERROR recursive delegation is not supported yet
}

reuse foo;
//~^ ERROR cycle detected when computing generics of `foo`

fn main() {}
