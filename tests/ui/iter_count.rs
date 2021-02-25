// run-rustfix
// aux-build:option_helpers.rs

#![warn(clippy::iter_count)]
#![allow(unused_variables)]
#![allow(unused_mut)]

extern crate option_helpers;

use option_helpers::IteratorFalsePositives;
use std::collections::{HashSet, VecDeque};

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

fn main() {
    let mut some_vec = vec![0, 1, 2, 3];
    let mut boxed_slice: Box<[u8]> = Box::new([0, 1, 2, 3]);
    let mut some_vec_deque: VecDeque<_> = some_vec.iter().cloned().collect();
    let mut some_hash_set = HashSet::new();
    some_hash_set.insert(1);

    {
        // Make sure we lint `.iter()` for relevant types.
        let bad_vec = some_vec.iter().count();
        let bad_slice = &some_vec[..].iter().count();
        let bad_boxed_slice = boxed_slice.iter().count();
        let bad_vec_deque = some_vec_deque.iter().count();
        let bad_hash_set = some_hash_set.iter().count();
    }

    {
        // Make sure we lint `.iter_mut()` for relevant types.
        let bad_vec = some_vec.iter_mut().count();
    }
    {
        let bad_slice = &some_vec[..].iter_mut().count();
    }
    {
        let bad_vec_deque = some_vec_deque.iter_mut().count();
    }

    // Make sure we don't lint for non-relevant types.
    let false_positive = HasIter;
    let ok = false_positive.iter().count();
    let ok_mut = false_positive.iter_mut().count();
}
