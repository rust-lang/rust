// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The Rust prelude
//!
//! Because `std` is required by most serious Rust software, it is
//! imported at the topmost level of every crate by default, as if the
//! first line of each crate was
//!
//! ```ignore
//! extern crate std;
//! ```
//!
//! This means that the contents of std can be accessed from any context
//! with the `std::` path prefix, as in `use std::vec`, `use std::task::spawn`,
//! etc.
//!
//! Additionally, `std` contains a `prelude` module that reexports many of the
//! most common traits, types and functions. The contents of the prelude are
//! imported into every *module* by default.  Implicitly, all modules behave as if
//! they contained the following prologue:
//!
//! ```ignore
//! use std::prelude::*;
//! ```
//!
//! The prelude is primarily concerned with exporting *traits* that are so
//! pervasive that it would be obnoxious to import for every use, particularly
//! those that define methods on primitive types. It does include a few
//! particularly useful standalone functions, like `from_str`, `range`, and
//! `drop`, `spawn`, and `channel`.

// Reexported core operators
pub use kinds::{Copy, Send, Sized, Share};
pub use ops::{Add, Sub, Mul, Div, Rem, Neg, Not};
pub use ops::{BitAnd, BitOr, BitXor};
pub use ops::{Drop, Deref, DerefMut};
pub use ops::{Shl, Shr, Index};
pub use option::{Option, Some, None};
pub use result::{Result, Ok, Err};

// Reexported functions
pub use from_str::from_str;
pub use iter::range;
pub use mem::drop;

// Reexported types and traits

pub use ascii::{Ascii, AsciiCast, OwnedAsciiCast, AsciiStr, IntoBytes};
pub use c_str::ToCStr;
pub use char::Char;
pub use clone::Clone;
pub use cmp::{Eq, Ord, TotalEq, TotalOrd, Ordering, Less, Equal, Greater, Equiv};
pub use container::{Container, Mutable, Map, MutableMap, Set, MutableSet};
pub use iter::{FromIterator, Extendable};
pub use iter::{Iterator, DoubleEndedIterator, RandomAccessIterator, CloneableIterator};
pub use iter::{OrdIterator, MutableDoubleEndedIterator, ExactSize};
pub use num::{Num, NumCast, CheckedAdd, CheckedSub, CheckedMul};
pub use num::{Signed, Unsigned};
pub use num::{Primitive, Int, Float, FloatMath, ToPrimitive, FromPrimitive};
pub use option::Expect;
pub use owned::Box;
pub use path::{GenericPath, Path, PosixPath, WindowsPath};
pub use ptr::RawPtr;
pub use io::{Buffer, Writer, Reader, Seek};
pub use str::{Str, StrVector, StrSlice, OwnedStr, IntoMaybeOwned};
pub use str::{StrAllocating};
pub use to_str::{ToStr, IntoStr};
pub use tuple::{Tuple1, Tuple2, Tuple3, Tuple4};
pub use tuple::{Tuple5, Tuple6, Tuple7, Tuple8};
pub use tuple::{Tuple9, Tuple10, Tuple11, Tuple12};
pub use slice::{CloneableVector, ImmutableCloneableVector, MutableCloneableVector};
pub use slice::{ImmutableVector, MutableVector};
pub use slice::{ImmutableEqVector, ImmutableTotalOrdVector, MutableTotalOrdVector};
pub use slice::{Vector, VectorVector, OwnedVector, MutableVectorAllocating};
pub use strbuf::StrBuf;
pub use vec::Vec;

// Reexported runtime types
pub use comm::{sync_channel, channel, SyncSender, Sender, Receiver};
pub use task::spawn;

// Reexported statics
#[cfg(not(test))]
pub use gc::GC;
