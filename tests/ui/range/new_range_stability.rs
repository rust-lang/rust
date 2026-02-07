// Stable

use std::range::{RangeInclusive, RangeInclusiveIter, RangeToInclusive};

fn range_inclusive(mut r: RangeInclusive<usize>) {
    &[1, 2, 3][r]; // Indexing

    r.start;
    r.last;
    r.contains(&5);
    r.is_empty();
    r.iter();

    let mut i = r.into_iter();
    i.next();
    i.remainder();
}

fn range_to_inclusive(mut r: RangeToInclusive<usize>) {
    &[1, 2, 3][r]; // Indexing

    r.last;
    r.contains(&5);
}

// Unstable module

use std::range::legacy; //~ ERROR unstable

// Unstable types

use std::range::RangeFrom; //~ ERROR unstable
use std::range::Range; //~ ERROR unstable
use std::range::RangeFromIter; //~ ERROR unstable
use std::range::RangeIter; //~ ERROR unstable

fn main() {}
