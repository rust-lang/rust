// Stable

use std::range::{
    RangeInclusive,
    RangeInclusiveIter,
    RangeToInclusive,
    RangeFrom,
    RangeFromIter,
    Range,
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
    i.remainder(); //~ ERROR unstable
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

fn range(mut r: Range<usize>) {
    &[1, 2, 3][r];

    r.start;
    r.end;
    r.contains(&5);
    r.is_empty();
    r.iter();

    let mut i = r.into_iter();
    i.next();

    // Left unstable
    i.remainder(); //~ ERROR unstable
}

// Unstable module

use std::range::legacy; //~ ERROR unstable

fn main() {}
