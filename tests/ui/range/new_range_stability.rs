// Stable

use std::range::{
    RangeInclusive,
    RangeInclusiveIter,
    RangeToInclusive,
    RangeFrom,
    RangeFromIter,
};

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

fn range_from(mut r: RangeFrom<usize>) {
    &[1, 2, 3][r]; // Indexing

    r.start;
    r.contains(&5);
    r.iter();

    let mut i = r.into_iter();
    i.next();

    // Left unstable
    i.remainder(); //~ ERROR unstable
}

// Unstable module

use std::range::legacy; //~ ERROR unstable

// Unstable types

use std::range::Range; //~ ERROR unstable
use std::range::RangeIter; //~ ERROR unstable

fn main() {}
