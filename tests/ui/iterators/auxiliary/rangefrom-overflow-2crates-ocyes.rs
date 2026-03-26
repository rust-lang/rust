//@ compile-flags: -C overflow-checks=yes

#![crate_type = "lib"]
#![feature(new_range_api)]

use std::range::RangeFromIter;

pub fn next(iter: &mut RangeFromIter<u8>) -> u8 {
    iter.next().unwrap()
}
