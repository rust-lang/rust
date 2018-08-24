// test that certain things are disallowed in constant functions

#![feature(const_fn, const_let)]

// no destructuring
const fn i((
            a,
            b
           ): (u32, u32)) -> u32 {
    a + b
}

fn main() {}
