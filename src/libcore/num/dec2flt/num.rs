// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Utility functions for bignums that don't make too much sense to turn into
//! methods.

// FIXME This module's name is a bit unfortunate, since other modules also
// import `core::num`.

use prelude::v1::*;
use cmp::Ordering::{self, Less, Equal, Greater};

pub use num::bignum::Big32x40 as Big;

/// Test whether truncating all bits less significant than `ones_place` introduces
/// a relative error less, equal, or greater than 0.5 ULP.
pub fn compare_with_half_ulp(f: &Big, ones_place: usize) -> Ordering {
    if ones_place == 0 {
        return Less;
    }
    let half_bit = ones_place - 1;
    if f.get_bit(half_bit) == 0 {
        // < 0.5 ULP
        return Less;
    }
    // If all remaining bits are zero, it's = 0.5 ULP, otherwise > 0.5
    // If there are no more bits (half_bit == 0), the below also correctly returns
    // Equal.
    for i in 0..half_bit {
        if f.get_bit(i) == 1 {
            return Greater;
        }
    }
    Equal
}

/// Convert an ASCII string containing only decimal digits to a `u64`.
///
/// Does not perform checks for overflow or invalid characters, so if the caller is not careful,
/// the result is bogus and can panic (though it won't be `unsafe`). Additionally, empty strings
/// are treated as zero. This function exists because
///
/// 1. using `FromStr` on `&[u8]` requires `from_utf8_unchecked`, which is bad, and
/// 2. piecing together the results of `integral.parse()` and `fractional.parse()` is
///    more complicated than this entire function.
pub fn from_str_unchecked<'a, T>(bytes: T) -> u64
    where T: IntoIterator<Item = &'a u8>
{
    let mut result = 0;
    for &c in bytes {
        result = result * 10 + (c - b'0') as u64;
    }
    result
}

/// Convert a string of ASCII digits into a bignum.
///
/// Like `from_str_unchecked`, this function relies on the parser to weed out non-digits.
pub fn digits_to_big(integral: &[u8], fractional: &[u8]) -> Big {
    let mut f = Big::from_small(0);
    for &c in integral.iter().chain(fractional) {
        let n = (c - b'0') as u32;
        f.mul_small(10);
        f.add_small(n);
    }
    f
}

/// Unwraps a bignum into a 64 bit integer. Panics if the number is too large.
pub fn to_u64(x: &Big) -> u64 {
    assert!(x.bit_length() < 64);
    let d = x.digits();
    if d.len() < 2 {
        d[0] as u64
    } else {
        (d[1] as u64) << 32 | d[0] as u64
    }
}


/// Extract a range of bits.

/// Index 0 is the least significant bit and the range is half-open as usual.
/// Panics if asked to extract more bits than fit into the return type.
pub fn get_bits(x: &Big, start: usize, end: usize) -> u64 {
    assert!(end - start <= 64);
    let mut result: u64 = 0;
    for i in (start..end).rev() {
        result = result << 1 | x.get_bit(i) as u64;
    }
    result
}
