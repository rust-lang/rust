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
pub use marker::{Copy, Send, Sized, Sync};
pub use ops::{Drop, Fn, FnMut, FnOnce, FullRange};

// Reexported functions
pub use iter::range;
pub use mem::drop;

// Reexported types and traits

pub use char::CharExt;
pub use clone::Clone;
pub use cmp::{PartialEq, PartialOrd, Eq, Ord};
pub use iter::{Extend, IteratorExt};
pub use iter::{Iterator, DoubleEndedIterator};
pub use iter::{ExactSizeIterator};
pub use option::Option::{self, Some, None};
pub use ptr::{PtrExt, MutPtrExt};
pub use result::Result::{self, Ok, Err};
pub use slice::{AsSlice, SliceExt};
pub use str::{Str, StrExt};
