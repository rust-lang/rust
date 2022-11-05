// compile-flags: -Z print-type-sizes
// build-pass
// ignore-pass

#![feature(start, generators, generator_trait)]

use std::ops::Generator;

fn generator<const C: usize>(array: [u8; C]) -> impl Generator<Yield = (), Return = ()> {
    move |()| {
        yield ();
        let _ = array;
    }
}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let _ = generator([0; 8192]);
    0
}
