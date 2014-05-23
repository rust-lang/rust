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
#[doc(noinline)] pub use kinds::{Copy, Send, Sized, Share};
#[doc(noinline)] pub use ops::{Add, Sub, Mul, Div, Rem, Neg, Not};
#[doc(noinline)] pub use ops::{BitAnd, BitOr, BitXor};
#[doc(noinline)] pub use ops::{Drop, Deref, DerefMut};
#[doc(noinline)] pub use ops::{Shl, Shr, Index};
#[doc(noinline)] pub use option::{Option, Some, None};
#[doc(noinline)] pub use result::{Result, Ok, Err};

// Reexported functions
#[doc(noinline)] pub use from_str::from_str;
#[doc(noinline)] pub use iter::range;
#[doc(noinline)] pub use mem::drop;

// Reexported types and traits

#[doc(noinline)] pub use ascii::{Ascii, AsciiCast, OwnedAsciiCast, AsciiStr};
#[doc(noinline)] pub use ascii::IntoBytes;
#[doc(noinline)] pub use c_str::ToCStr;
#[doc(noinline)] pub use char::Char;
#[doc(noinline)] pub use clone::Clone;
#[doc(noinline)] pub use cmp::{Eq, Ord, TotalEq, TotalOrd};
#[doc(noinline)] pub use cmp::{Ordering, Less, Equal, Greater, Equiv};
#[doc(noinline)] pub use container::{Container, Mutable, Map, MutableMap};
#[doc(noinline)] pub use container::{Set, MutableSet};
#[doc(noinline)] pub use iter::{FromIterator, Extendable, ExactSize};
#[doc(noinline)] pub use iter::{Iterator, DoubleEndedIterator};
#[doc(noinline)] pub use iter::{RandomAccessIterator, CloneableIterator};
#[doc(noinline)] pub use iter::{OrdIterator, MutableDoubleEndedIterator};
#[doc(noinline)] pub use num::{Num, NumCast, CheckedAdd, CheckedSub, CheckedMul};
#[doc(noinline)] pub use num::{Signed, Unsigned, Primitive, Int, Float};
#[doc(noinline)] pub use num::{FloatMath, ToPrimitive, FromPrimitive};
#[doc(noinline)] pub use option::Expect;
#[doc(noinline)] pub use owned::Box;
#[doc(noinline)] pub use path::{GenericPath, Path, PosixPath, WindowsPath};
#[doc(noinline)] pub use ptr::RawPtr;
#[doc(noinline)] pub use io::{Buffer, Writer, Reader, Seek};
#[doc(noinline)] pub use str::{Str, StrVector, StrSlice, OwnedStr};
#[doc(noinline)] pub use str::{IntoMaybeOwned, StrAllocating};
#[doc(noinline)] pub use to_str::{ToStr, IntoStr};
#[doc(noinline)] pub use tuple::{Tuple1, Tuple2, Tuple3, Tuple4};
#[doc(noinline)] pub use tuple::{Tuple5, Tuple6, Tuple7, Tuple8};
#[doc(noinline)] pub use tuple::{Tuple9, Tuple10, Tuple11, Tuple12};
#[doc(noinline)] pub use slice::{CloneableVector, ImmutableCloneableVector};
#[doc(noinline)] pub use slice::{MutableCloneableVector, MutableTotalOrdVector};
#[doc(noinline)] pub use slice::{ImmutableVector, MutableVector};
#[doc(noinline)] pub use slice::{ImmutableEqVector, ImmutableTotalOrdVector};
#[doc(noinline)] pub use slice::{Vector, VectorVector, OwnedVector};
#[doc(noinline)] pub use slice::MutableVectorAllocating;
#[doc(noinline)] pub use string::String;
#[doc(noinline)] pub use vec::Vec;

// Reexported runtime types
#[doc(noinline)] pub use comm::{sync_channel, channel};
#[doc(noinline)] pub use comm::{SyncSender, Sender, Receiver};
#[doc(noinline)] pub use task::spawn;

// Reexported statics
#[cfg(not(test))]
#[doc(noinline)] pub use gc::GC;
