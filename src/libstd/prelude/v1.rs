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

#![stable(feature = "rust1", since = "1.0.0")]

// Reexported core operators
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use marker::{Copy, Send, Sized, Sync};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use ops::{Drop, Fn, FnMut, FnOnce};

// Reexported functions
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use mem::drop;

// Reexported types and traits
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use boxed::Box;
#[cfg(stage0)]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use char::CharExt;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use clone::Clone;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use cmp::{PartialEq, PartialOrd, Eq, Ord};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use iter::DoubleEndedIterator;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use iter::ExactSizeIterator;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use iter::{Iterator, IteratorExt, Extend};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use option::Option::{self, Some, None};
#[cfg(stage0)]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use ptr::{PtrExt, MutPtrExt};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use result::Result::{self, Ok, Err};
#[cfg(stage0)]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use slice::{SliceExt, SliceConcatExt, AsSlice};
#[cfg(not(stage0))]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use slice::{SliceConcatExt, AsSlice};
#[cfg(stage0)]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use str::{Str, StrExt};
#[cfg(not(stage0))]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use str::Str;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use string::{String, ToString};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use vec::Vec;

// NB: remove when path reform lands
#[doc(no_inline)] pub use old_path::{Path, GenericPath};
// NB: remove when I/O reform lands
#[doc(no_inline)] pub use old_io::{Buffer, Writer, Reader, Seek, BufferPrelude};
// NB: remove when range syntax lands
#[allow(deprecated)]
#[doc(no_inline)] pub use iter::range;

#[doc(no_inline)] pub use num::wrapping::{Wrapping, WrappingOps};
