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
//! A quick summary:
//!
//! Implementations of the following traits:
//!
//! * `FromStr`
//! * `Not`
//! * `Ord`
//! * `TotalOrd`
//! * `Eq`
//! * `Default`
//! * `Zero`
//!
//! A `to_bit` conversion function.

use from_str::FromStr;
use num::{Int, one, zero};
use option::{None, Option, Some};

#[cfg(not(test))] use cmp::{Eq, Ord, TotalOrd, Ordering};
#[cfg(not(test))] use ops::{Not, BitAnd, BitOr, BitXor};
#[cfg(not(test))] use default::Default;

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

/////////////////////////////////////////////////////////////////////////////
// Trait impls on `bool`
/////////////////////////////////////////////////////////////////////////////

impl FromStr for bool {
    /// Parse a `bool` from a string.
    ///
    /// Yields an `Option<bool>`, because `s` may or may not actually be parseable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(from_str::<bool>("true"), Some(true));
    /// assert_eq!(from_str::<bool>("false"), Some(false));
    /// assert_eq!(from_str::<bool>("not even a boolean"), None);
    /// ```
    #[inline]
    fn from_str(s: &str) -> Option<bool> {
        match s {
            "true"  => Some(true),
            "false" => Some(false),
            _       => None,
        }
    }
}

#[cfg(not(test))]
impl Not<bool> for bool {
    /// The logical complement of a boolean value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(!true, false);
    /// assert_eq!(!false, true);
    /// ```
    #[inline]
    fn not(&self) -> bool { !*self }
}

#[cfg(not(test))]
impl BitAnd<bool, bool> for bool {
    /// Conjunction of two boolean values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(false.bitand(&false), false);
    /// assert_eq!(true.bitand(&false), false);
    /// assert_eq!(false.bitand(&true), false);
    /// assert_eq!(true.bitand(&true), true);
    ///
    /// assert_eq!(false & false, false);
    /// assert_eq!(true & false, false);
    /// assert_eq!(false & true, false);
    /// assert_eq!(true & true, true);
    /// ```
    #[inline]
    fn bitand(&self, b: &bool) -> bool { *self & *b }
}

#[cfg(not(test))]
impl BitOr<bool, bool> for bool {
    /// Disjunction of two boolean values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(false.bitor(&false), false);
    /// assert_eq!(true.bitor(&false), true);
    /// assert_eq!(false.bitor(&true), true);
    /// assert_eq!(true.bitor(&true), true);
    ///
    /// assert_eq!(false | false, false);
    /// assert_eq!(true | false, true);
    /// assert_eq!(false | true, true);
    /// assert_eq!(true | true, true);
    /// ```
    #[inline]
    fn bitor(&self, b: &bool) -> bool { *self | *b }
}

#[cfg(not(test))]
impl BitXor<bool, bool> for bool {
    /// An 'exclusive or' of two boolean values.
    ///
    /// 'exclusive or' is identical to `or(and(a, not(b)), and(not(a), b))`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(false.bitxor(&false), false);
    /// assert_eq!(true.bitxor(&false), true);
    /// assert_eq!(false.bitxor(&true), true);
    /// assert_eq!(true.bitxor(&true), false);
    ///
    /// assert_eq!(false ^ false, false);
    /// assert_eq!(true ^ false, true);
    /// assert_eq!(false ^ true, true);
    /// assert_eq!(true ^ true, false);
    /// ```
    #[inline]
    fn bitxor(&self, b: &bool) -> bool { *self ^ *b }
}

#[cfg(not(test))]
impl Ord for bool {
    #[inline]
    fn lt(&self, other: &bool) -> bool {
        to_bit::<u8>(*self) < to_bit(*other)
    }
}

#[cfg(not(test))]
impl TotalOrd for bool {
    #[inline]
    fn cmp(&self, other: &bool) -> Ordering {
        to_bit::<u8>(*self).cmp(&to_bit(*other))
    }
}

/// Equality between two boolean values.
///
/// Two booleans are equal if they have the same value.
///
/// # Examples
///
/// ```rust
/// assert_eq!(false.eq(&true), false);
/// assert_eq!(false == false, true);
/// assert_eq!(false != true, true);
/// assert_eq!(false.ne(&false), false);
/// ```
#[cfg(not(test))]
impl Eq for bool {
    #[inline]
    fn eq(&self, other: &bool) -> bool { (*self) == (*other) }
}

#[cfg(not(test))]
impl Default for bool {
    fn default() -> bool { false }
}

#[cfg(test)]
mod tests {
    use prelude::*;
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
    fn test_from_str() {
        assert_eq!(from_str::<bool>("true"), Some(true));
        assert_eq!(from_str::<bool>("false"), Some(false));
        assert_eq!(from_str::<bool>("not even a boolean"), None);
    }

    #[test]
    fn test_to_str() {
        assert_eq!(false.to_str(), ~"false");
        assert_eq!(true.to_str(), ~"true");
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
