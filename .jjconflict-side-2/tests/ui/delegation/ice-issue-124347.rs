#![feature(fn_delegation)]

trait Trait {
    reuse Trait::foo { &self.0 }
    //~^ ERROR failed to resolve delegation callee
    //~| ERROR: this function takes 0 arguments but 1 argument was supplied
}

reuse foo;
//~^ ERROR failed to resolve delegation callee
//~| WARN: function cannot return without recursing

fn main() {}
