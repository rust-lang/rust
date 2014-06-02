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
#[doc(no_inline)] pub use kinds::{Copy, Send, Sized, Share};
#[doc(no_inline)] pub use ops::{Add, Sub, Mul, Div, Rem, Neg, Not};
#[doc(no_inline)] pub use ops::{BitAnd, BitOr, BitXor};
#[doc(no_inline)] pub use ops::{Drop, Deref, DerefMut};
#[doc(no_inline)] pub use ops::{Shl, Shr, Index};
#[doc(no_inline)] pub use option::{Option, Some, None};
#[doc(no_inline)] pub use result::{Result, Ok, Err};

// Reexported functions
#[doc(no_inline)] pub use from_str::from_str;
#[doc(no_inline)] pub use iter::range;
#[doc(no_inline)] pub use mem::drop;

// Reexported types and traits

#[doc(no_inline)] pub use ascii::{Ascii, AsciiCast, OwnedAsciiCast, AsciiStr};
#[doc(no_inline)] pub use ascii::IntoBytes;
#[doc(no_inline)] pub use c_str::ToCStr;
#[doc(no_inline)] pub use char::Char;
#[doc(no_inline)] pub use clone::Clone;
#[doc(no_inline)] pub use cmp::{PartialEq, PartialOrd, Eq, Ord};
#[doc(no_inline)] pub use cmp::{Ordering, Less, Equal, Greater, Equiv};
#[doc(no_inline)] pub use container::{Container, Mutable, Map, MutableMap};
#[doc(no_inline)] pub use container::{Set, MutableSet};
#[doc(no_inline)] pub use iter::{FromIterator, Extendable, ExactSize};
#[doc(no_inline)] pub use iter::{Iterator, DoubleEndedIterator};
#[doc(no_inline)] pub use iter::{RandomAccessIterator, CloneableIterator};
#[doc(no_inline)] pub use iter::{OrdIterator, MutableDoubleEndedIterator};
#[doc(no_inline)] pub use num::{Num, NumCast, CheckedAdd, CheckedSub, CheckedMul};
#[doc(no_inline)] pub use num::{Signed, Unsigned, Primitive, Int, Float};
#[doc(no_inline)] pub use num::{FloatMath, ToPrimitive, FromPrimitive};
#[doc(no_inline)] pub use owned::Box;
#[doc(no_inline)] pub use path::{GenericPath, Path, PosixPath, WindowsPath};
#[doc(no_inline)] pub use ptr::RawPtr;
#[doc(no_inline)] pub use io::{Buffer, Writer, Reader, Seek};
#[doc(no_inline)] pub use str::{Str, StrVector, StrSlice, OwnedStr};
#[doc(no_inline)] pub use str::{IntoMaybeOwned, StrAllocating};
#[doc(no_inline)] pub use to_str::{ToStr, IntoStr};
#[doc(no_inline)] pub use tuple::{Tuple1, Tuple2, Tuple3, Tuple4};
#[doc(no_inline)] pub use tuple::{Tuple5, Tuple6, Tuple7, Tuple8};
#[doc(no_inline)] pub use tuple::{Tuple9, Tuple10, Tuple11, Tuple12};
#[doc(no_inline)] pub use slice::{CloneableVector, ImmutableCloneableVector};
#[doc(no_inline)] pub use slice::{MutableCloneableVector, MutableOrdVector};
#[doc(no_inline)] pub use slice::{ImmutableVector, MutableVector};
#[doc(no_inline)] pub use slice::{ImmutableEqVector, ImmutableOrdVector};
#[doc(no_inline)] pub use slice::{Vector, VectorVector, OwnedVector};
#[doc(no_inline)] pub use slice::MutableVectorAllocating;
#[doc(no_inline)] pub use string::String;
#[doc(no_inline)] pub use vec::Vec;

// Reexported runtime types
#[doc(no_inline)] pub use comm::{sync_channel, channel};
#[doc(no_inline)] pub use comm::{SyncSender, Sender, Receiver};
#[doc(no_inline)] pub use task::spawn;

// Reexported statics
#[cfg(not(test))]
#[doc(no_inline)] pub use gc::GC;
