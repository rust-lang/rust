#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait<'a, 'b, 'c, A = usize, B = u32, C = String, const N: usize = 123> {
    fn foo<T = usize>(&self) {
        //~^ ERROR: defaults for generic parameters are not allowed here
        //~| WARN: this was previously accepted by the compiler but is being phased out
    }
}

reuse Trait::foo;

fn main() {}
