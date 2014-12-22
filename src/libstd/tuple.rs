// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations on tuples
//!
//! To access a single element of a tuple one can use the following
//! methods:
//!
//! * `valN` - returns a value of _N_-th element
//! * `refN` - returns a reference to _N_-th element
//! * `mutN` - returns a mutable reference to _N_-th element
//!
//! Indexing starts from zero, so `val0` returns first value, `val1`
//! returns second value, and so on. In general, a tuple with _S_
//! elements provides aforementioned methods suffixed with numbers
//! from `0` to `S-1`. Traits which contain these methods are
//! implemented for tuples with up to 12 elements.
//!
//! If every type inside a tuple implements one of the following
//! traits, then a tuple itself also implements it.
//!
//! * `Clone`
//! * `PartialEq`
//! * `Eq`
//! * `PartialOrd`
//! * `Ord`
//! * `Default`
//!
//! # Examples
//!
//! Using methods:
//!
//! ```
//! #[allow(deprecated)]
//! # fn main() {
//! let pair = ("pi", 3.14f64);
//! assert_eq!(pair.val0(), "pi");
//! assert_eq!(pair.val1(), 3.14f64);
//! # }
//! ```
//!
//! Using traits implemented for tuples:
//!
//! ```
//! use std::default::Default;
//!
//! let a = (1i, 2i);
//! let b = (3i, 4i);
//! assert!(a != b);
//!
//! let c = b.clone();
//! assert!(b == c);
//!
//! let d : (u32, f32) = Default::default();
//! assert_eq!(d, (0u32, 0.0f32));
//! ```

#![doc(primitive = "tuple")]
#![stable]
