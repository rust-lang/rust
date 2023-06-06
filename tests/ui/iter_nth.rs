//@aux-build:option_helpers.rs

#![warn(clippy::iter_nth)]
#![allow(clippy::useless_vec)]

#[macro_use]
extern crate option_helpers;

use option_helpers::IteratorFalsePositives;
use std::collections::VecDeque;

/// Struct to generate false positives for things with `.iter()`.
#[derive(Copy, Clone)]
struct HasIter;

impl HasIter {
    fn iter(self) -> IteratorFalsePositives {
        IteratorFalsePositives { foo: 0 }
    }

    fn iter_mut(self) -> IteratorFalsePositives {
        IteratorFalsePositives { foo: 0 }
    }
}

/// Checks implementation of `ITER_NTH` lint.
fn iter_nth() {
    let mut some_vec = vec![0, 1, 2, 3];
    let mut boxed_slice: Box<[u8]> = Box::new([0, 1, 2, 3]);
    let mut some_vec_deque: VecDeque<_> = some_vec.iter().cloned().collect();

    {
        // Make sure we lint `.iter()` for relevant types.
        let bad_vec = some_vec.iter().nth(3);
        let bad_slice = &some_vec[..].iter().nth(3);
        let bad_boxed_slice = boxed_slice.iter().nth(3);
        let bad_vec_deque = some_vec_deque.iter().nth(3);
    }

    {
        // Make sure we lint `.iter_mut()` for relevant types.
        let bad_vec = some_vec.iter_mut().nth(3);
    }
    {
        let bad_slice = &some_vec[..].iter_mut().nth(3);
    }
    {
        let bad_vec_deque = some_vec_deque.iter_mut().nth(3);
    }

    // Make sure we don't lint for non-relevant types.
    let false_positive = HasIter;
    let ok = false_positive.iter().nth(3);
    let ok_mut = false_positive.iter_mut().nth(3);
}

fn main() {}
