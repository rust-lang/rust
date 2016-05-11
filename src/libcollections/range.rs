// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "collections_range",
            reason = "waiting for dust to settle on inclusive ranges",
            issue = "30877")]

//! Range syntax.

use core::option::Option::{self, None, Some};
use core::ops::{RangeFull, Range, RangeTo, RangeFrom, RangeInclusive, RangeToInclusive};

/// **RangeArgument** is implemented by Rust's built-in range types, produced
/// by range syntax like `..`, `a..`, `..b` or `c..d`.
pub trait RangeArgument<T> {
    /// Start index (inclusive)
    ///
    /// Return start value if present, else `None`.
    fn start(&self) -> Option<T> {
        None
    }

    /// End index (exclusive)
    ///
    /// Return end value if present, else `None`.
    fn end(&self) -> Option<T> {
        None
    }
}

impl<T> RangeArgument<T> for RangeFull {}

impl<T: Copy> RangeArgument<T> for RangeFrom<T> {
    fn start(&self) -> Option<T> {
        Some(self.start)
    }
}

impl<T: Copy> RangeArgument<T> for RangeTo<T> {
    fn end(&self) -> Option<T> {
        Some(self.end)
    }
}

impl<T: Copy> RangeArgument<T> for Range<T> {
    fn start(&self) -> Option<T> {
        Some(self.start)
    }
    fn end(&self) -> Option<T> {
        Some(self.end)
    }
}

macro_rules! inclusive {
    ($Int: ty) => {
        impl RangeArgument<$Int> for RangeToInclusive<$Int> {
            fn end(&self) -> Option<$Int> {
                Some(self.end.checked_add(1).expect("inclusive range to maximum usize"))
            }
        }

        impl RangeArgument<$Int> for RangeInclusive<$Int> {
            fn start(&self) -> Option<$Int> {
                match *self {
                    RangeInclusive::Empty { at } => Some(at),
                    RangeInclusive::NonEmpty { start, .. } => Some(start),
                }
            }
            fn end(&self) -> Option<$Int> {
                match *self {
                    RangeInclusive::Empty { at } => Some(at),
                    RangeInclusive::NonEmpty { end, .. } => {
                        Some(end.checked_add(1).expect("inclusive range to maximum usize"))
                    }
                }
            }
        }
    }
}

inclusive!(u8);
inclusive!(u16);
inclusive!(u32);
inclusive!(u64);
inclusive!(usize);
inclusive!(i8);
inclusive!(i16);
inclusive!(i32);
inclusive!(i64);
inclusive!(isize);
