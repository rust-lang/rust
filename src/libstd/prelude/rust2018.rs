// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! The 2018 edition of the prelude of The Rust Standard Library.
//!
//! See the [module-level documentation](../index.html) for more.


#![unstable(feature = "rust2018_prelude", issue = "51418")]

// Re-exported core operators
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use marker::{Copy, Send, Sized, Sync};
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use ops::{Drop, Fn, FnMut, FnOnce};

// Re-exported functions
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use mem::drop;

// Re-exported types and traits
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use clone::Clone;
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use cmp::{PartialEq, PartialOrd, Eq, Ord};
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use convert::{AsRef, AsMut, Into, From};
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use default::Default;
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use iter::{Iterator, Extend, IntoIterator};
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use iter::{DoubleEndedIterator, ExactSizeIterator};
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use option::Option::{self, Some, None};
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use result::Result::{self, Ok, Err};


// Contents so far are equivalent to src/libcore/prelude/v1.rs


// Re-exported types and traits that involve memory allocation
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use boxed::Box;
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use borrow::ToOwned;
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use slice::SliceConcatExt;
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use string::{String, ToString};
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use vec::Vec;


// Contents so far are equivalent to v1.rs


// Not in v1.rs because of breakage: https://github.com/rust-lang/rust/pull/49518
#[unstable(feature = "rust2018_prelude", issue = "51418")]
#[doc(no_inline)] pub use convert::{TryFrom, TryInto};
