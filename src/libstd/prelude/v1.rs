// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The first version of the prelude of the standard library.

#![stable]

// Reexported core operators
#[stable] #[doc(no_inline)] pub use marker::{Copy, Send, Sized, Sync};
#[stable] #[doc(no_inline)] pub use ops::{Drop, Fn, FnMut, FnOnce};

// TEMPORARY
#[unstable] #[doc(no_inline)] pub use ops::FullRange;

// Reexported functions
#[stable] #[doc(no_inline)] pub use mem::drop;

// Reexported types and traits

#[stable] #[doc(no_inline)] pub use boxed::Box;
#[stable] #[doc(no_inline)] pub use char::CharExt;
#[stable] #[doc(no_inline)] pub use clone::Clone;
#[stable] #[doc(no_inline)] pub use cmp::{PartialEq, PartialOrd, Eq, Ord};
#[stable] #[doc(no_inline)] pub use iter::DoubleEndedIterator;
#[stable] #[doc(no_inline)] pub use iter::ExactSizeIterator;
#[stable] #[doc(no_inline)] pub use iter::{Iterator, IteratorExt, Extend};
#[stable] #[doc(no_inline)] pub use option::Option::{self, Some, None};
#[stable] #[doc(no_inline)] pub use ptr::{PtrExt, MutPtrExt};
#[stable] #[doc(no_inline)] pub use result::Result::{self, Ok, Err};
#[stable] #[doc(no_inline)] pub use slice::AsSlice;
#[stable] #[doc(no_inline)] pub use slice::{SliceExt, SliceConcatExt};
#[stable] #[doc(no_inline)] pub use str::{Str, StrExt};
#[stable] #[doc(no_inline)] pub use string::{String, ToString};
#[stable] #[doc(no_inline)] pub use vec::Vec;

// NB: remove when path reform lands
#[doc(no_inline)] pub use path::{Path, GenericPath};
// NB: remove when I/O reform lands
#[doc(no_inline)] pub use io::{Buffer, Writer, Reader, Seek, BufferPrelude};
// NB: remove when range syntax lands
#[doc(no_inline)] pub use iter::range;
