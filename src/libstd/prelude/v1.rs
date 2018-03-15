// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The first version of the prelude of The Rust Standard Library.
//!
//! See the [module-level documentation](../index.html) for more.

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
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use borrow::ToOwned;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use clone::Clone;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use cmp::{PartialEq, PartialOrd, Eq, Ord};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use convert::{AsRef, AsMut, Into, From};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use default::Default;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use iter::{Iterator, Extend, IntoIterator};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use iter::{DoubleEndedIterator, ExactSizeIterator};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use option::Option::{self, Some, None};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use result::Result::{self, Ok, Err};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use slice::SliceConcatExt;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use string::{String, ToString};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)] pub use vec::Vec;
