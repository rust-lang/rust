// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:--test

/// ```rust,edition2018
/// #![feature(try_blocks)]
///
/// use std::num::ParseIntError;
///
/// let result: Result<i32, ParseIntError> = try {
///     "1".parse::<i32>()?
///         + "2".parse::<i32>()?
///         + "3".parse::<i32>()?
/// };
/// assert_eq!(result, Ok(6));
///
/// let result: Result<i32, ParseIntError> = try {
///     "1".parse::<i32>()?
///         + "foo".parse::<i32>()?
///         + "3".parse::<i32>()?
/// };
/// assert!(result.is_err());
/// ```


/// ```rust,edition2015,compile_fail,E0574
/// #![feature(try_blocks)]
///
/// use std::num::ParseIntError;
///
/// let result: Result<i32, ParseIntError> = try {
///     "1".parse::<i32>()?
///         + "2".parse::<i32>()?
///         + "3".parse::<i32>()?
/// };
/// assert_eq!(result, Ok(6));
///
/// let result: Result<i32, ParseIntError> = try {
///     "1".parse::<i32>()?
///         + "foo".parse::<i32>()?
///         + "3".parse::<i32>()?
/// };
/// assert!(result.is_err());
/// ```

pub fn foo() {}
