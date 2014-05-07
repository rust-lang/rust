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
//! For more information, see std::prelude.

// Reexported core operators
pub use kinds::{Copy, Send, Sized, Share};
pub use ops::{Add, Sub, Mul, Div, Rem, Neg, Not};
pub use ops::{BitAnd, BitOr, BitXor};
pub use ops::{Drop, Deref, DerefMut};
pub use ops::{Shl, Shr, Index};
pub use option::{Option, Some, None};
pub use result::{Result, Ok, Err};

// Reexported functions
pub use iter::range;
pub use mem::drop;

// Reexported types and traits

pub use char::Char;
pub use clone::Clone;
pub use cmp::{Eq, Ord, TotalEq, TotalOrd, Ordering, Less, Equal, Greater, Equiv};
pub use container::{Container, Mutable, Map, MutableMap, Set, MutableSet};
pub use iter::{FromIterator, Extendable};
pub use iter::{Iterator, DoubleEndedIterator, RandomAccessIterator, CloneableIterator};
pub use iter::{OrdIterator, MutableDoubleEndedIterator, ExactSize};
pub use num::{Num, NumCast, CheckedAdd, CheckedSub, CheckedMul};
pub use num::{Signed, Unsigned};
pub use num::{Primitive, Int, ToPrimitive, FromPrimitive};
pub use ptr::RawPtr;
pub use str::{Str, StrSlice};
pub use tuple::{Tuple1, Tuple2, Tuple3, Tuple4};
pub use tuple::{Tuple5, Tuple6, Tuple7, Tuple8};
pub use tuple::{Tuple9, Tuple10, Tuple11, Tuple12};
pub use slice::{ImmutableEqVector, ImmutableTotalOrdVector};
pub use slice::{MutableVector};
pub use slice::{Vector, ImmutableVector};
