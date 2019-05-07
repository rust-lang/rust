#![feature(generators, generator_trait)]

use std::ops::Generator;

fn foo() -> impl Generator<Return = i32> {
    || {
        if false {
            return Ok(6); //~ ERROR mismatched types [E0308]
        }

        yield ();

        5
    }
}

fn main() {}
