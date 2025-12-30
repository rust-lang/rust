// Stable

use std::range::{RangeInclusive, RangeInclusiveIter};

fn range_inclusive(mut r: RangeInclusive<usize>) {
    r.start;
    r.last;
    r.contains(&5);
    r.is_empty();
    r.iter();

    let mut i = r.into_iter();
    i.next();
    i.remainder();
}

// Unstable module

use std::range::legacy; //~ ERROR unstable

// Unstable types

use std::range::RangeFrom; //~ ERROR unstable
use std::range::Range; //~ ERROR unstable
use std::range::RangeFromIter; //~ ERROR unstable
use std::range::RangeIter; //~ ERROR unstable

fn main() {}
