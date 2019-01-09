// run-pass

#![feature(generators, generator_trait)]

use std::ops::{GeneratorState, Generator};

fn finish<T>(mut amt: usize, mut t: T) -> T::Return
    where T: Generator<Yield = ()>
{
    loop {
        match unsafe { t.resume() } {
            GeneratorState::Yielded(()) => amt = amt.checked_sub(1).unwrap(),
            GeneratorState::Complete(ret) => {
                assert_eq!(amt, 0);
                return ret
            }
        }
    }

}

fn main() {
    finish(1, || yield);
    finish(8, || {
        for _ in 0..8 {
            yield;
        }
    });
    finish(1, || {
        if true {
            yield;
        } else {
        }
    });
    finish(1, || {
        if false {
        } else {
            yield;
        }
    });
    finish(2, || {
        if { yield; false } {
            yield;
            panic!()
        }
        yield
    });
}
