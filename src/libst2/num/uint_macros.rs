// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![experimental]
#![macro_escape]
#![doc(hidden)]
#![allow(unsigned_negation)]

macro_rules! uint_module (($T:ty) => (

// String conversion functions and impl num -> str

/// Convert to a string as a byte slice in a given base.
///
/// Use in place of x.to_string() when you do not need to store the string permanently
///
/// # Examples
///
/// ```
/// #![allow(deprecated)]
///
/// std::uint::to_str_bytes(123, 10, |v| {
///     assert!(v == "123".as_bytes());
/// });
/// ```
#[inline]
#[deprecated = "just use .to_string(), or a BufWriter with write! if you mustn't allocate"]
pub fn to_str_bytes<U>(n: $T, radix: uint, f: |v: &[u8]| -> U) -> U { unimplemented!() }

))
