// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:--test

// This file tests the source-partitioning behavior of rustdoc.
// Each test contains some code that should be put into the generated
// `fn main` and some attributes should be left outside (except the first
// one, which has no attributes).
// If the #![recursion_limit] attribute is incorrectly left inside,
// then the tests will fail because the macro recurses 128 times.

/// ```
/// assert_eq!(1 + 1, 2);
/// ```
pub fn simple() {}

/// ```
/// #![recursion_limit = "1024"]
/// macro_rules! recurse {
///     (()) => {};
///     (() $($rest:tt)*) => { recurse!($($rest)*); }
/// }
/// recurse!(() () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ());
/// assert_eq!(1 + 1, 2);
/// ```
pub fn non_feature_attr() {}

/// ```
/// #![feature(core_intrinsics)]
/// assert_eq!(1 + 1, 2);
/// ```
pub fn feature_attr() {}

/// ```
/// #![feature(core_intrinsics)]
/// #![recursion_limit = "1024"]
/// macro_rules! recurse {
///     (()) => {};
///     (() $($rest:tt)*) => { recurse!($($rest)*); }
/// }
/// recurse!(() () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ());
/// assert_eq!(1 + 1, 2);
/// ```
pub fn both_attrs() {}

/// ```
/// #![recursion_limit = "1024"]
/// #![feature(core_intrinsics)]
/// macro_rules! recurse {
///     (()) => {};
///     (() $($rest:tt)*) => { recurse!($($rest)*); }
/// }
/// recurse!(() () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ()
///          () () () () () () () ());
/// assert_eq!(1 + 1, 2);
/// ```
pub fn both_attrs_reverse() {}

