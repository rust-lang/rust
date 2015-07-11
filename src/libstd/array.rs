// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A fixed-size array is denoted `[T; N]` for the element type `T` and
//! the compile time constant size `N`. The size must be zero or positive.
//!
//! Arrays values are created either with an explicit expression that lists
//! each element: `[x, y, z]` or a repeat expression: `[x; N]`. The repeat
//! expression requires that the element type is `Copy`.
//!
//! The type `[T; N]` is `Copy` if `T: Copy`.
//!
//! Arrays of sizes from 0 to 32 (inclusive) implement the following traits
//! if the element type allows it:
//!
//! - `Clone`
//! - `Debug`
//! - `IntoIterator` (implemented for `&[T; N]` and `&mut [T; N]`)
//! - `PartialEq`, `PartialOrd`, `Ord`, `Eq`
//! - `Hash`
//! - `AsRef`, `AsMut`
//!
//! Arrays dereference to [slices (`[T]`)][slice], so their methods can be called
//! on arrays.
//!
//! [slice]: primitive.slice.html
//!
//! Rust does not currently support generics over the size of an array type.
//!
//! # Examples
//!
//! ```
//! let mut array: [i32; 3] = [0; 3];
//!
//! array[1] = 1;
//! array[2] = 2;
//!
//! assert_eq!([1, 2], &array[1..]);
//!
//! // This loop prints: 0 1 2
//! for x in &array {
//!     print!("{} ", x);
//! }
//!
//! ```
//!

#![doc(primitive = "array")]
