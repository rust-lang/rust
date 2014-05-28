// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![macro_escape]
#![doc(hidden)]

macro_rules! uint_module (($T:ty, $T_SIGNED:ty, $bits:expr) => (

pub static BITS : uint = $bits;
pub static BYTES : uint = ($bits / 8);

pub static MIN: $T = 0 as $T;
pub static MAX: $T = 0 as $T - 1 as $T;

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;

    use num;
    use num::CheckedDiv;
    use num::Bitwise;

    #[test]
    fn test_overflows() {
        assert!(MAX > 0);
        assert!(MIN <= 0);
        assert!(MIN + MAX + 1 == 0);
    }

    #[test]
    fn test_num() {
        num::test_num(10 as $T, 2 as $T);
    }

    #[test]
    fn test_bitwise() {
        assert!(0b1110 as $T == (0b1100 as $T).bitor(&(0b1010 as $T)));
        assert!(0b1000 as $T == (0b1100 as $T).bitand(&(0b1010 as $T)));
        assert!(0b0110 as $T == (0b1100 as $T).bitxor(&(0b1010 as $T)));
        assert!(0b1110 as $T == (0b0111 as $T).shl(&(1 as $T)));
        assert!(0b0111 as $T == (0b1110 as $T).shr(&(1 as $T)));
        assert!(MAX - (0b1011 as $T) == (0b1011 as $T).not());
    }

    #[test]
    fn test_count_ones() {
        assert!((0b0101100 as $T).count_ones() == 3);
        assert!((0b0100001 as $T).count_ones() == 2);
        assert!((0b1111001 as $T).count_ones() == 5);
    }

    #[test]
    fn test_count_zeros() {
        assert!((0b0101100 as $T).count_zeros() == BITS as $T - 3);
        assert!((0b0100001 as $T).count_zeros() == BITS as $T - 2);
        assert!((0b1111001 as $T).count_zeros() == BITS as $T - 5);
    }

    #[test]
    fn test_unsigned_checked_div() {
        assert!(10u.checked_div(&2) == Some(5));
        assert!(5u.checked_div(&0) == None);
    }
}

))
