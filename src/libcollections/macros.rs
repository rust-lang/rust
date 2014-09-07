// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Creates a `Vec` containing the arguments.
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
macro_rules! vec {
    ($x:expr; $y:expr) => (
        <[_] as $crate::slice::SliceExt>::into_vec(
            $crate::boxed::Box::new([$x; $y]))
    );
    ($($x:expr),*) => (
        <[_] as $crate::slice::SliceExt>::into_vec(
            $crate::boxed::Box::new([$($x),*]))
    );
    ($($x:expr,)*) => (vec![$($x),*])
}

/// Use the syntax described in `std::fmt` to create a value of type `String`.
/// See `std::fmt` for more information.
///
/// # Example
///
/// ```
/// format!("test");
/// format!("hello {}", "world!");
/// format!("x = {}, y = {y}", 10, y = 30);
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
macro_rules! format {
    ($($arg:tt)*) => ($crate::fmt::format(format_args!($($arg)*)))
}
