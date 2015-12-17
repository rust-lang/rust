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
//! Rust comes with a variety of things in its standard library. However, if
//! you had to manually import every single thing that you used, it would be
//! very verbose. But importing a lot of things that a program never uses isn't
//! good either. A balance needs to be struck.
//!
//! The *prelude* is the list of things that Rust automatically imports into
//! every Rust program. It's kept as small as possible, and is focused on
//! things, particuarly traits, which are used in almost every single Rust
//! program.
//!
//! On a technical level, Rust inserts
//!
//! ```ignore
//! extern crate std;
//! ```
//!
//! into the crate root of every crate, and
//!
//! ```ignore
//! use std::prelude::v1::*;
//! ```
//!
//! into every module.
//!
//! # Other preludes
//!
//! Preludes can be seen as a pattern to make using multiple types more
//! convenient. As such, you'll find other preludes in the standard library,
//! such as [`std::io::prelude`]. Various libraries in the Rust ecosystem may
//! also define their own preludes.
//!
//! [`std::io::prelude`]: ../io/prelude/index.html
//!
//! The difference between 'the prelude' and these other preludes is that they
//! are not automatically `use`'d, and must be imported manually. This is still
//! easier than importing all of their consitutent components.
//!
//! # Prelude contents
//!
//! The current version of the prelude (version 1) lives in
//! [`std::prelude::v1`], and reexports the following.
//!
//! * [`std::marker`]::{[`Copy`], [`Send`], [`Sized`], [`Sync`]}. The marker
//!   traits indicate fundamental properties of types.
//! * [`std::ops`]::{[`Drop`], [`Fn`], [`FnMut`], [`FnOnce`]}. Various
//!   operations for both destuctors and overloading `()`.
//! * [`std::mem`]::[`drop`], a convenience function for explicitly dropping a
//!   value.
//! * [`std::boxed`]::[`Box`], a way to allocate values on the heap.
//! * [`std::borrow`]::[`ToOwned`], The conversion trait that defines
//!   [`to_owned()`], the generic method for creating an owned type from a
//!   borrowed type.
//! * [`std::clone`]::[`Clone`], the ubiquitous trait that defines [`clone()`],
//!   the method for producing a copy of a value.
//! * [`std::cmp`]::{[`PartialEq`], [`PartialOrd`], [`Eq`], [`Ord`] }. The
//!   comparison traits, which implement the comparison operators and are often
//!   seen in trait bounds.
//! * [`std::convert`]::{[`AsRef`], [`AsMut`], [`Into`], [`From`]}. Generic
//!   conversions, used by savvy API authors to create overloaded methods.
//! * [`std::default`]::[`Default`], types that have default values.
//! * [`std::iter`]::{[`Iterator`], [`Extend`], [`IntoIterator`],
//!   [`DoubleEndedIterator`], [`ExactSizeIterator`]}. Iterators of various
//!   kinds.
//! * [`std::option`]::[`Option`]::{`self`, `Some`, `None`}. A type which
//!   expresses the presence or absence of a value. This type is so commonly
//!   used, its variants are also exported.
//! * [`std::result`]::[`Result`]::{`self`, `Ok`, `Err`}. A type for functions
//!   that may succeed or fail. Like [`Option`], its variants are exported as
//!   well.
//! * [`std::slice`]::[`SliceConcatExt`], a trait that exists for technical
//!   reasons, but shouldn't have to exist. It provides a few useful methods on
//!   slices.
//! * [`std::string`]::{[`String`], [`ToString`]}, heap allocated strings.
//! * [`std::vec`]::[`Vec`](../vec/struct.Vec.html), a growable, heap-allocated
//!   vector.
//!
//! [`AsMut`]: ../convert/trait.AsMut.html
//! [`AsRef`]: ../convert/trait.AsRef.html
//! [`Box`]: ../boxed/struct.Box.html
//! [`Clone`]: ../clone/trait.Clone.html
//! [`Copy`]: ../marker/trait.Copy.html
//! [`Default`]: ../default/trait.Default.html
//! [`DoubleEndedIterator`]: ../iter/trait.DoubleEndedIterator.html
//! [`Drop`]: ../ops/trait.Drop.html
//! [`Eq`]: ../cmp/trait.Eq.html
//! [`ExactSizeIterator`]: ../iter/trait.ExactSizeIterator.html
//! [`Extend`]: ../iter/trait.Extend.html
//! [`FnMut`]: ../ops/trait.FnMut.html
//! [`FnOnce`]: ../ops/trait.FnOnce.html
//! [`Fn`]: ../ops/trait.Fn.html
//! [`From`]: ../convert/trait.From.html
//! [`IntoIterator`]: ../iter/trait.IntoIterator.html
//! [`Into`]: ../convert/trait.Into.html
//! [`Iterator`]: ../iter/trait.Iterator.html
//! [`Option`]: ../option/enum.Option.html
//! [`Ord`]: ../cmp/trait.Ord.html
//! [`PartialEq`]: ../cmp/trait.PartialEq.html
//! [`PartialOrd`]: ../cmp/trait.PartialOrd.html
//! [`Result`]: ../result/enum.Result.html
//! [`Send`]: ../marker/trait.Send.html
//! [`Sized`]: ../marker/trait.Sized.html
//! [`SliceConcatExt`]: ../slice/trait.SliceConcatExt.html
//! [`String`]: ../string/struct.String.html
//! [`Sync`]: ../marker/trait.Sync.html
//! [`ToOwned`]: ../borrow/trait.ToOwned.html
//! [`ToString`]: ../string/trait.ToString.html
//! [`Vec`]: ../vec/struct.Vec.html
//! [`clone()`]: ../clone/trait.Clone.html#tymethod.clone
//! [`drop`]: ../mem/fn.drop.html
//! [`std::borrow`]: ../borrow/index.html
//! [`std::boxed`]: ../boxed/index.html
//! [`std::clone`]: ../clone/index.html
//! [`std::cmp`]: ../cmp/index.html
//! [`std::convert`]: ../convert/index.html
//! [`std::default`]: ../default/index.html
//! [`std::iter`]: ../iter/index.html
//! [`std::marker`]: ../marker/index.html
//! [`std::mem`]: ../mem/index.html
//! [`std::ops`]: ../ops/index.html
//! [`std::option`]: ../option/index.html
//! [`std::prelude::v1`]: v1/index.html
//! [`std::result`]: ../result/index.html
//! [`std::slice`]: ../slice/index.html
//! [`std::string`]: ../string/index.html
//! [`std::vec`]: ../vec/index.html
//! [`to_owned()`]: ../borrow/trait.ToOwned.html#tymethod.to_owned
//! [book-closures]: ../../book/closures.html
//! [book-dtor]: ../../book/drop.html
//! [book-enums]: ../../book/enums.html
//! [book-iter]: ../../book/iterators.html

#![stable(feature = "rust1", since = "1.0.0")]

pub mod v1;
