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
//! well. This module is imported by default when `#![no_std]` is used in the
//! same manner as the standard library's prelude.

#![stable(feature = "core_prelude", since = "1.4.0")]

// Reexported core operators
#[doc(no_inline)] pub use marker::{Copy, Send, Sized, Sync};
#[doc(no_inline)] pub use ops::{Drop, Fn, FnMut, FnOnce};

// Reexported functions
#[doc(no_inline)] pub use mem::drop;

// Reexported types and traits
#[doc(no_inline)] pub use clone::Clone;
#[doc(no_inline)] pub use cmp::{PartialEq, PartialOrd, Eq, Ord};
#[doc(no_inline)] pub use convert::{AsRef, AsMut, Into, From};
#[doc(no_inline)] pub use default::Default;
#[doc(no_inline)] pub use iter::{Iterator, Extend, IntoIterator};
#[doc(no_inline)] pub use iter::{DoubleEndedIterator, ExactSizeIterator};
#[doc(no_inline)] pub use option::Option::{self, Some, None};
#[doc(no_inline)] pub use result::Result::{self, Ok, Err};

// Reexported extension traits for primitive types
#[doc(no_inline)] pub use slice::SliceExt;
#[doc(no_inline)] pub use str::StrExt;
#[doc(no_inline)] pub use char::CharExt;
