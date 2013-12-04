// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Many programming languages have a 'prelude': a particular subset of the
libraries that come with the language. Every program imports the prelude by
default.

For example, it would be annoying to add `use std::io::println;` to every single
program, and the vast majority of Rust programs will wish to print to standard
output. Therefore, it makes sense to import it into every program.

Rust's prelude has three main parts:

1. io::print and io::println.
2. Core operators, such as `Add`, `Mul`, and `Not`.
3. Various types and traits, such as `Clone`, `Eq`, and `comm::Chan`.

*/

// Reexported core operators
pub use either::{Either, Left, Right};
pub use kinds::Sized;
pub use kinds::{Freeze, Send};
pub use ops::{Add, Sub, Mul, Div, Rem, Neg, Not};
pub use ops::{BitAnd, BitOr, BitXor};
pub use ops::{Drop};
pub use ops::{Shl, Shr, Index};
pub use option::{Option, Some, None};
pub use result::{Result, Ok, Err};

// Reexported functions
pub use from_str::from_str;
pub use iter::range;
pub use io::stdio::{print, println};

// Reexported types and traits

pub use any::{Any, AnyOwnExt, AnyRefExt, AnyMutRefExt};
pub use ascii::{Ascii, AsciiCast, OwnedAsciiCast, AsciiStr, ToBytesConsume};
pub use bool::Bool;
pub use c_str::ToCStr;
pub use char::Char;
pub use clone::{Clone, DeepClone};
pub use cmp::{Eq, ApproxEq, Ord, TotalEq, TotalOrd, Ordering, Less, Equal, Greater, Equiv};
pub use container::{Container, Mutable, Map, MutableMap, Set, MutableSet};
pub use default::Default;
pub use from_str::FromStr;
pub use hash::Hash;
pub use iter::{FromIterator, Extendable};
pub use iter::{Iterator, DoubleEndedIterator, RandomAccessIterator, ClonableIterator};
pub use iter::{OrdIterator, MutableDoubleEndedIterator, ExactSize};
pub use num::Times;
pub use num::{Algebraic, Trigonometric, Exponential, Hyperbolic};
pub use num::{Bitwise, BitCount, Bounded};
pub use num::{Integer, Fractional, Real, RealExt};
pub use num::{Num, NumCast, CheckedAdd, CheckedSub, CheckedMul};
pub use num::{Orderable, Signed, Unsigned, Round};
pub use num::{Primitive, Int, Float, ToStrRadix, ToPrimitive, FromPrimitive};
pub use path::{GenericPath, Path, PosixPath, WindowsPath};
pub use ptr::RawPtr;
pub use io::{Buffer, Writer, Reader, Seek};
pub use send_str::{SendStr, SendStrOwned, SendStrStatic, IntoSendStr};
pub use str::{Str, StrVector, StrSlice, OwnedStr};
pub use to_bytes::IterBytes;
pub use to_str::{ToStr, ToStrConsume};
pub use tuple::{CopyableTuple, ImmutableTuple};
pub use tuple::{ImmutableTuple1, ImmutableTuple2, ImmutableTuple3, ImmutableTuple4};
pub use tuple::{ImmutableTuple5, ImmutableTuple6, ImmutableTuple7, ImmutableTuple8};
pub use tuple::{ImmutableTuple9, ImmutableTuple10, ImmutableTuple11, ImmutableTuple12};
pub use tuple::{Tuple1, Tuple2, Tuple3, Tuple4};
pub use tuple::{Tuple5, Tuple6, Tuple7, Tuple8};
pub use tuple::{Tuple9, Tuple10, Tuple11, Tuple12};
pub use vec::{ImmutableEqVector, ImmutableTotalOrdVector, ImmutableCopyableVector};
pub use vec::{OwnedVector, OwnedCopyableVector,OwnedEqVector, MutableVector};
pub use vec::{Vector, VectorVector, CopyableVector, ImmutableVector};

// Reexported runtime types
pub use comm::{stream, Port, Chan, GenericChan, GenericSmartChan, GenericPort, Peekable};
pub use task::spawn;

/// Disposes of a value.
#[inline]
pub fn drop<T>(_x: T) { }
