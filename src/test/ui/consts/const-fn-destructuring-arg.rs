// test that certain things are disallowed in constant functions

#![feature(const_fn)]

// no destructuring
const fn i((
            a,
            //~^ ERROR arguments of constant functions can only be immutable by-value bindings
            b
            //~^ ERROR arguments of constant functions can only be immutable by-value bindings
           ): (u32, u32)) -> u32 {
    a + b
    //~^ ERROR let bindings in constant functions are unstable
    //~| ERROR let bindings in constant functions are unstable
}

fn main() {}
