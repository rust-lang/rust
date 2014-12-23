// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The core prelude
//!
//! This module is intended for users of libcore which do not link to libstd as
//! well. This module is not imported by default, but using the entire contents
//! of this module will provide all of the useful traits and types in libcore
//! that one would expect from the standard library as well.
//!
//! There is no method to automatically inject this prelude, and this prelude is
//! a subset of the standard library's prelude.
//!
//! # Example
//!
//! ```ignore
//! # fn main() {
//! #![feature(globs)]
//!
//! use core::prelude::*;
//! # }
//! ```

// Reexported core operators
pub use kinds::{Copy, Send, Sized, Sync};
pub use ops::{Add, Sub, Mul, Div, Rem, Neg, Not};
pub use ops::{BitAnd, BitOr, BitXor};
pub use ops::{Drop, Deref, DerefMut};
pub use ops::{Shl, Shr};
pub use ops::{Index, IndexMut};
pub use ops::{Slice, SliceMut};
pub use ops::{Fn, FnMut, FnOnce};

// Reexported functions
pub use iter::range;
pub use mem::drop;
pub use str::from_str;

// Reexported types and traits

pub use char::Char;
pub use clone::Clone;
pub use cmp::{PartialEq, PartialOrd, Eq, Ord};
pub use cmp::{Ordering, Equiv};
pub use cmp::Ordering::{Less, Equal, Greater};
pub use iter::{FromIterator, Extend, IteratorExt};
pub use iter::{Iterator, DoubleEndedIterator, DoubleEndedIteratorExt, RandomAccessIterator};
pub use iter::{IteratorCloneExt, CloneIteratorExt};
pub use iter::{IteratorOrdExt, MutableDoubleEndedIterator, ExactSizeIterator};
pub use num::{ToPrimitive, FromPrimitive};
pub use option::Option;
pub use option::Option::{Some, None};
pub use ptr::RawPtr;
pub use result::Result;
pub use result::Result::{Ok, Err};
pub use str::{Str, StrExt};
pub use tuple::{Tuple1, Tuple2, Tuple3, Tuple4};
pub use tuple::{Tuple5, Tuple6, Tuple7, Tuple8};
pub use tuple::{Tuple9, Tuple10, Tuple11, Tuple12};
pub use slice::{PartialEqSliceExt, OrdSliceExt};
pub use slice::{AsSlice, SliceExt};
