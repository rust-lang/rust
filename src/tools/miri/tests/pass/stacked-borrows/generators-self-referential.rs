// See https://github.com/rust-lang/unsafe-code-guidelines/issues/148:
// this fails when Stacked Borrows is strictly applied even to `!Unpin` types.
#![feature(generators, generator_trait)]

use std::{
    ops::{Generator, GeneratorState},
    pin::Pin,
};

fn firstn() -> impl Generator<Yield = u64, Return = ()> {
    static move || {
        let mut num = 0;
        let num = &mut num;

        yield *num;
        *num += 1; // would fail here

        yield *num;
        *num += 1;

        yield *num;
        *num += 1;
    }
}

fn main() {
    let mut generator_iterator = firstn();
    let mut pin = unsafe { Pin::new_unchecked(&mut generator_iterator) };
    let mut sum = 0;
    while let GeneratorState::Yielded(x) = pin.as_mut().resume(()) {
        sum += x;
    }
    assert_eq!(sum, 3);
}
