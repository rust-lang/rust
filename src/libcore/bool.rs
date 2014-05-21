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
//! * `Not`
//! * `BitAnd`
//! * `BitOr`
//! * `BitXor`
//! * `Ord`
//! * `TotalOrd`
//! * `Eq`
//! * `TotalEq`
//! * `Default`
//!

#[cfg(not(test))] use cmp::{Eq, Ord, TotalOrd, Ordering, TotalEq, Less, Equal, Greater};
#[cfg(not(test))] use ops::{Not};
#[cfg(not(test))] use default::Default;

/////////////////////////////////////////////////////////////////////////////
// Trait impls on `bool`
/////////////////////////////////////////////////////////////////////////////

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
impl Ord for bool {
    #[inline]
    fn lt(&self, other: &bool) -> bool {
        *self == false && *other != false
    }
}

#[cfg(not(test))]
impl TotalOrd for bool {
    #[inline]
    fn cmp(&self, other: &bool) -> Ordering {
        if *self == *other {
            Equal
        } else if *self == false {
            Less
        } else {
            Greater
        }
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
impl TotalEq for bool {}

#[cfg(not(test))]
impl Default for bool {
    fn default() -> bool { false }
}

#[cfg(test)]
mod tests {
    use realstd::prelude::*;

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
        assert_eq!(false.to_str(), "false".to_owned());
        assert_eq!(true.to_str(), "true".to_owned());
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
