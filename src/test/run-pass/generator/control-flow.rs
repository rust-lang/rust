// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
