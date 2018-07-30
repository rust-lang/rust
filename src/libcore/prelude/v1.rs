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

// Re-exported core operators
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use marker::{Copy, Send, Sized, Sync};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use ops::{Drop, Fn, FnMut, FnOnce};

// Re-exported functions
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use mem::drop;

// Re-exported types and traits
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use clone::Clone;
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use cmp::{PartialEq, PartialOrd, Eq, Ord};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use convert::{AsRef, AsMut, Into, From};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use default::Default;
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use iter::{Iterator, Extend, IntoIterator};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use iter::{DoubleEndedIterator, ExactSizeIterator};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use option::Option::{self, Some, None};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use result::Result::{self, Ok, Err};
