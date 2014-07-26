// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![doc(primitive = "unit")]
#![unstable = "this module is purely for documentation and it will likely be \
               removed from the public api"]

//! The `()` type, sometimes called "unit" or "nil".
//!
//! The `()` type has exactly one value `()`, and is used when there
//! is no other meaningful value that could be returned. `()` is most
//! commonly seen implicitly: functions without a `-> ...` implicitly
//! have return type `()`, that is, these are equivalent:
//!
//! ```rust
//! fn long() -> () {}
//!
//! fn short() {}
//! ```
//!
//! The semicolon `;` can be used to discard the result of an
//! expression at the end of a block, making the expression (and thus
//! the block) evaluate to `()`. For example,
//!
//! ```rust
//! fn returns_i64() -> i64 {
//!     1i64
//! }
//! fn returns_unit() {
//!     1i64;
//! }
//!
//! let is_i64 = {
//!     returns_i64()
//! };
//! let is_unit = {
//!     returns_i64();
//! };
//! ```
