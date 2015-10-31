// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The Rust Prelude
//!
//! Because `std` is required by most serious Rust software, it is
//! imported at the topmost level of every crate by default, as if
//! each crate contains the following:
//!
//! ```ignore
//! extern crate std;
//! ```
//!
//! This means that the contents of std can be accessed from any context
//! with the `std::` path prefix, as in `use std::vec`, `use std::thread::spawn`,
//! etc.
//!
//! Additionally, `std` contains a versioned *prelude* that reexports many of the
//! most common traits, types, and functions. *The contents of the prelude are
//! imported into every module by default*.  Implicitly, all modules behave as if
//! they contained the following [`use` statement][book-use]:
//!
//! [book-use]: ../../book/crates-and-modules.html#importing-modules-with-use
//!
//! ```ignore
//! use std::prelude::v1::*;
//! ```
//!
//! The prelude is primarily concerned with exporting *traits* that
//! are so pervasive that they would be onerous to import for every use,
//! particularly those that are commonly mentioned in [generic type
//! bounds][book-traits].
//!
//! The current version of the prelude (version 1) lives in
//! [`std::prelude::v1`](v1/index.html), and reexports the following.
//!
//! * `std::marker::`{
//!     [`Copy`](../marker/trait.Copy.html),
//!     [`Send`](../marker/trait.Send.html),
//!     [`Sized`](../marker/trait.Sized.html),
//!     [`Sync`](../marker/trait.Sync.html)
//!   }.
//!   The marker traits indicate fundamental properties of types.
//! * `std::ops::`{
//!     [`Drop`](../ops/trait.Drop.html),
//!     [`Fn`](../ops/trait.Fn.html),
//!     [`FnMut`](../ops/trait.FnMut.html),
//!     [`FnOnce`](../ops/trait.FnOnce.html)
//!   }.
//!   The [destructor][book-dtor] trait and the
//!   [closure][book-closures] traits, reexported from the same
//!   [module that also defines overloaded
//!   operators](../ops/index.html).
//! * `std::mem::`[`drop`](../mem/fn.drop.html).
//!   A convenience function for explicitly dropping a value.
//! * `std::boxed::`[`Box`](../boxed/struct.Box.html).
//!   The owned heap pointer.
//! * `std::borrow::`[`ToOwned`](../borrow/trait.ToOwned.html).
//!   The conversion trait that defines `to_owned`, the generic method
//!   for creating an owned type from a borrowed type.
//! * `std::clone::`[`Clone`](../clone/trait.Clone.html).
//!   The ubiquitous trait that defines `clone`, the method for
//!   producing copies of values that are consider expensive to copy.
//! * `std::cmp::`{
//!     [`PartialEq`](../cmp/trait.PartialEq.html),
//!     [`PartialOrd`](../cmp/trait.PartialOrd.html),
//!     [`Eq`](../cmp/trait.Eq.html),
//!     [`Ord`](../cmp/trait.Ord.html)
//!   }.
//!   The comparison traits, which implement the comparison operators
//!   and are often seen in trait bounds.
//! * `std::convert::`{
//!     [`AsRef`](../convert/trait.AsRef.html),
//!     [`AsMut`](../convert/trait.AsMut.html),
//!     [`Into`](../convert/trait.Into.html),
//!     [`From`](../convert/trait.From.html)
//!   }.
//!   Generic conversions, used by savvy API authors to create
//!   overloaded methods.
//! * `std::default::`[`Default`](../default/trait.Default.html).
//!   Types that have default values.
//! * `std::iter::`{
//!     [`Iterator`](../iter/trait.Iterator.html),
//!     [`Extend`](../iter/trait.Extend.html),
//!     [`IntoIterator`](../iter/trait.IntoIterator.html),
//!     [`DoubleEndedIterator`](../iter/trait.DoubleEndedIterator.html),
//!     [`ExactSizeIterator`](../iter/trait.ExactSizeIterator.html)
//!   }.
//!   [Iterators][book-iter].
//! * `std::option::Option::`{
//!     [`self`](../option/enum.Option.html),
//!     [`Some`](../option/enum.Option.html),
//!     [`None`](../option/enum.Option.html)
//!   }.
//!   The ubiquitous `Option` type and its two [variants][book-enums],
//!   `Some` and `None`.
//! * `std::result::Result::`{
//!     [`self`](../result/enum.Result.html),
//!     [`Ok`](../result/enum.Result.html),
//!     [`Err`](../result/enum.Result.html)
//!   }.
//!   The ubiquitous `Result` type and its two [variants][book-enums],
//!   `Ok` and `Err`.
//! * `std::slice::`[`SliceConcatExt`](../slice/trait.SliceConcatExt.html).
//!   An unstable extension to slices that shouldn't have to exist.
//! * `std::string::`{
//!     [`String`](../string/struct.String.html),
//!     [`ToString`](../string/trait.ToString.html)
//!   }.
//!   Heap allocated strings.
//! * `std::vec::`[`Vec`](../vec/struct.Vec.html).
//!   Heap allocated vectors.
//!
//! [book-traits]: ../../book/traits.html
//! [book-closures]: ../../book/closures.html
//! [book-dtor]: ../../book/drop.html
//! [book-iter]: ../../book/iterators.html
//! [book-enums]: ../../book/enums.html

#![stable(feature = "rust1", since = "1.0.0")]

pub mod v1;
