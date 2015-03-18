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
///
/// `vec!` allows `Vec`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a `Vec` containing a given list of elements:
///
/// ```
/// let v = vec![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// ```
///
/// - Create a `Vec` from a given element and size:
///
/// ```
/// let v = vec![1; 3];
/// assert_eq!(v, [1, 1, 1]);
/// ```
///
/// Note that unlike array expressions this syntax supports all elements
/// which implement `Clone` and the number of elements doesn't have to be
/// a constant.
#[cfg(not(test))]
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
macro_rules! vec {
    ($elem:expr; $n:expr) => (
        $crate::vec::from_elem($elem, $n)
    );
    ($($x:expr),*) => (
        <[_]>::into_vec($crate::boxed::Box::new([$($x),*]))
    );
    ($($x:expr,)*) => (vec![$($x),*])
}

// HACK(japaric): with cfg(test) the inherent `[T]::into_vec` method, which is
// required for this macro definition, is not available. Instead use the
// `slice::into_vec`  function which is only available with cfg(test)
// NB see the slice::hack module in slice.rs for more information
#[cfg(test)]
macro_rules! vec {
    ($elem:expr; $n:expr) => (
        $crate::vec::from_elem($elem, $n)
    );
    ($($x:expr),*) => (
        $crate::slice::into_vec($crate::boxed::Box::new([$($x),*]))
    );
    ($($x:expr,)*) => (vec![$($x),*])
}

/// Use the syntax described in `std::fmt` to create a value of type `String`.
/// See `std::fmt` for more information.
///
/// # Examples
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
