// compile-flags: -Z print-type-sizes --crate-type=lib
// build-pass
// ignore-pass

#![feature(generators, generator_trait)]

use std::ops::Generator;

fn generator<const C: usize>(array: [u8; C]) -> impl Generator<Yield = (), Return = ()> {
    move |()| {
        yield ();
        let _ = array;
    }
}

pub fn foo() {
    let _ = generator([0; 8192]);
}
