#![feature(generators, generator_trait)]

use std::ops::Generator;

fn msg() -> u32 {
    0
}

pub fn foo() -> impl Generator<Yield=(), Return=u32> {
    || {
        yield;
        return msg();
    }
}
