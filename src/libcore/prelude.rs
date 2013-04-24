// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The Rust prelude. Imported into every module by default.

/* Reexported core operators */

pub use either::{Either, Left, Right};
pub use kinds::{Const, Copy, Owned, Durable};
#[cfg(stage0)]
pub use ops::{Add, Sub, Mul, Div, Modulo, Neg, Not};
#[cfg(stage1)]
#[cfg(stage2)]
#[cfg(stage3)]
pub use ops::{Add, Sub, Mul, Quot, Rem, Neg, Not};
pub use ops::{BitAnd, BitOr, BitXor};
pub use ops::{Drop};
pub use ops::{Shl, Shr, Index};
pub use option::{Option, Some, None};
pub use result::{Result, Ok, Err};

/* Reexported functions */

pub use io::{print, println};

/* Reexported types and traits */

pub use clone::Clone;
pub use cmp::{Eq, Ord, TotalEq, TotalOrd, Ordering, Less, Equal, Greater};
pub use container::{Container, Mutable, Map, Set};
pub use hash::Hash;
pub use iter::{BaseIter, ReverseIter, MutableIter, ExtendedIter, EqIter};
pub use iter::{CopyableIter, CopyableOrderedIter, CopyableNonstrictIter};
pub use iter::{Times, ExtendedMutableIter};
pub use num::{Num, Signed, Unsigned, Natural, NumCast};
pub use path::GenericPath;
pub use path::Path;
pub use path::PosixPath;
pub use path::WindowsPath;
pub use ptr::Ptr;
pub use ascii::{Ascii, AsciiCast, OwnedAsciiCast, AsciiStr};
pub use str::{StrSlice, OwnedStr};
pub use to_bytes::IterBytes;
pub use to_str::{ToStr, ToStrConsume};
pub use tuple::{CopyableTuple, ImmutableTuple, ExtendedTupleOps};
pub use vec::{CopyableVector, ImmutableVector};
pub use vec::{ImmutableEqVector, ImmutableCopyableVector};
pub use vec::{OwnedVector, OwnedCopyableVector, MutableVector};
pub use io::{Reader, ReaderUtil, Writer, WriterUtil};

/* Reexported runtime types */
pub use comm::{stream, Port, Chan, GenericChan, GenericSmartChan, GenericPort, Peekable};
pub use task::spawn;

/* Reexported modules */

pub use at_vec;
pub use bool;
pub use cast;
pub use char;
pub use cmp;
pub use either;
pub use f32;
pub use f64;
pub use float;
pub use i16;
pub use i32;
pub use i64;
pub use i8;
pub use int;
pub use io;
pub use iter;
pub use libc;
pub use num;
pub use ops;
pub use option;
pub use os;
pub use path;
pub use comm;
pub use unstable;
pub use ptr;
pub use rand;
pub use result;
pub use str;
pub use sys;
pub use task;
pub use to_str;
pub use u16;
pub use u32;
pub use u64;
pub use u8;
pub use uint;
pub use vec;
