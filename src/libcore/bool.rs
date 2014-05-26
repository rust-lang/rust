// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations on boolean values (`bool` type)
//!
//! A `to_bit` conversion function.

use num::{Int, one, zero};

/////////////////////////////////////////////////////////////////////////////
// Freestanding functions
/////////////////////////////////////////////////////////////////////////////

/// Convert a `bool` to an integer.
///
/// # Examples
///
/// ```rust
/// use std::bool;
///
/// assert_eq!(bool::to_bit::<u8>(true), 1u8);
/// assert_eq!(bool::to_bit::<u8>(false), 0u8);
/// ```
#[inline]
pub fn to_bit<N: Int>(p: bool) -> N {
    if p { one() } else { zero() }
}

#[cfg(test)]
mod tests {
    use realstd::prelude::*;
    use super::to_bit;

    #[test]
    fn test_to_bit() {
        assert_eq!(to_bit::<u8>(true), 1u8);
        assert_eq!(to_bit::<u8>(false), 0u8);
    }

    #[test]
    fn test_eq() {
        assert_eq!(false.eq(&true), false);
        assert_eq!(false == false, true);
        assert_eq!(false != true, true);
        assert_eq!(false.ne(&false), false);
    }

    #[test]
    fn test_bitand() {
        assert_eq!(false.bitand(&false), false);
        assert_eq!(true.bitand(&false), false);
        assert_eq!(false.bitand(&true), false);
        assert_eq!(true.bitand(&true), true);

        assert_eq!(false & false, false);
        assert_eq!(true & false, false);
        assert_eq!(false & true, false);
        assert_eq!(true & true, true);
    }

    #[test]
    fn test_bitor() {
        assert_eq!(false.bitor(&false), false);
        assert_eq!(true.bitor(&false), true);
        assert_eq!(false.bitor(&true), true);
        assert_eq!(true.bitor(&true), true);

        assert_eq!(false | false, false);
        assert_eq!(true | false, true);
        assert_eq!(false | true, true);
        assert_eq!(true | true, true);
    }

    #[test]
    fn test_bitxor() {
        assert_eq!(false.bitxor(&false), false);
        assert_eq!(true.bitxor(&false), true);
        assert_eq!(false.bitxor(&true), true);
        assert_eq!(true.bitxor(&true), false);

        assert_eq!(false ^ false, false);
        assert_eq!(true ^ false, true);
        assert_eq!(false ^ true, true);
        assert_eq!(true ^ true, false);
    }

    #[test]
    fn test_not() {
        assert_eq!(!true, false);
        assert_eq!(!false, true);
    }

    #[test]
    fn test_to_str() {
        let s = false.to_str();
        assert_eq!(s.as_slice(), "false");
        let s = true.to_str();
        assert_eq!(s.as_slice(), "true");
    }

    #[test]
    fn test_ord() {
        assert!(true > false);
        assert!(!(false > true));

        assert!(false < true);
        assert!(!(true < false));

        assert!(false <= false);
        assert!(false >= false);
        assert!(true <= true);
        assert!(true >= true);

        assert!(false <= true);
        assert!(!(false >= true));
        assert!(true >= false);
        assert!(!(true <= false));
    }

    #[test]
    fn test_totalord() {
        assert!(true.cmp(&true) == Equal);
        assert!(false.cmp(&false) == Equal);
        assert!(true.cmp(&false) == Greater);
        assert!(false.cmp(&true) == Less);
    }
}
