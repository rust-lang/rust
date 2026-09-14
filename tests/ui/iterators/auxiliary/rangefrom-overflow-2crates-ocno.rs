//@ compile-flags: -C overflow-checks=no

#![crate_type = "lib"]

use std::range::RangeFromIter;

pub fn next(iter: &mut RangeFromIter<u8>) -> u8 {
    iter.next().unwrap()
}
