// Regression test for incorrect DropAndReplace behavior introduced in #60840
// and fixed in #61373. When combined with the optimization implemented in
// #60187, this produced incorrect code for generators when a saved local was
// re-assigned.

#![feature(generators, generator_trait)]

use std::ops::{Generator, GeneratorState};
use std::pin::Pin;

#[derive(Debug, PartialEq)]
struct Foo(i32);

impl Drop for Foo {
    fn drop(&mut self) { }
}

fn main() {
    let mut a = || {
        let mut x = Foo(4);
        yield;
        assert_eq!(x.0, 4);

        // At one point this tricked our dataflow analysis into thinking `x` was
        // StorageDead after the assignment.
        x = Foo(5);
        assert_eq!(x.0, 5);

        {
            let y = Foo(6);
            yield;
            assert_eq!(y.0, 6);
        }

        assert_eq!(x.0, 5);
    };

    loop {
        match Pin::new(&mut a).resume() {
            GeneratorState::Complete(()) => break,
            _ => (),
        }
    }
}
