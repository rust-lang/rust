//! The Rust Prelude.
//!
//! Rust comes with a variety of things in its standard library. However, if
//! you had to manually import every single thing that you used, it would be
//! very verbose. But importing a lot of things that a program never uses isn't
//! good either. A balance needs to be struck.
//!
//! The *prelude* is the list of things that Rust automatically imports into
//! every Rust program. It's kept as small as possible, and is focused on
//! things, particularly traits, which are used in almost every single Rust
//! program.
//!
//! # Other preludes
//!
//! Preludes can be seen as a pattern to make using multiple types more
//! convenient. As such, you'll find other preludes in the standard library,
//! such as [`std::io::prelude`]. Various libraries in the Rust ecosystem may
//! also define their own preludes.
//!
//! [`std::io::prelude`]: crate::io::prelude
//!
//! The difference between 'the prelude' and these other preludes is that they
//! are not automatically `use`'d, and must be imported manually. This is still
//! easier than importing all of their constituent components.
//!
//! # Prelude contents
//!
//! The current version of the prelude (version 1) lives in
//! [`std::prelude::v1`], and re-exports the following:
//!
//! * [`std::marker`]::{[`Copy`], [`Send`], [`Sized`], [`Sync`], [`Unpin`]},
//!   marker traits that indicate fundamental properties of types.
//! * [`std::ops`]::{[`Drop`], [`Fn`], [`FnMut`], [`FnOnce`]}, various
//!   operations for both destructors and overloading `()`.
//! * [`std::mem`]::[`drop`][`mem::drop`], a convenience function for explicitly
//!   dropping a value.
//! * [`std::boxed`]::[`Box`], a way to allocate values on the heap.
//! * [`std::borrow`]::[`ToOwned`], the conversion trait that defines
//!   [`to_owned`], the generic method for creating an owned type from a
//!   borrowed type.
//! * [`std::clone`]::[`Clone`], the ubiquitous trait that defines
//!   [`clone`][`Clone::clone`], the method for producing a copy of a value.
//! * [`std::cmp`]::{[`PartialEq`], [`PartialOrd`], [`Eq`], [`Ord`] }, the
//!   comparison traits, which implement the comparison operators and are often
//!   seen in trait bounds.
//! * [`std::convert`]::{[`AsRef`], [`AsMut`], [`Into`], [`From`]}, generic
//!   conversions, used by savvy API authors to create overloaded methods.
//! * [`std::default`]::[`Default`], types that have default values.
//! * [`std::iter`]::{[`Iterator`], [`Extend`], [`IntoIterator`]
//!   [`DoubleEndedIterator`], [`ExactSizeIterator`]}, iterators of various
//!   kinds.
//! * [`std::option`]::[`Option`]::{[`self`][`Option`], [`Some`], [`None`]}, a
//!   type which expresses the presence or absence of a value. This type is so
//!   commonly used, its variants are also exported.
//! * [`std::result`]::[`Result`]::{[`self`][`Result`], [`Ok`], [`Err`]}, a type
//!   for functions that may succeed or fail. Like [`Option`], its variants are
//!   exported as well.
//! * [`std::string`]::{[`String`], [`ToString`]}, heap allocated strings.
//! * [`std::vec`]::[`Vec`], a growable, heap-allocated vector.
//!
//! [`mem::drop`]: crate::mem::drop
//! [`std::borrow`]: crate::borrow
//! [`std::boxed`]: crate::boxed
//! [`std::clone`]: crate::clone
//! [`std::cmp`]: crate::cmp
//! [`std::convert`]: crate::convert
//! [`std::default`]: crate::default
//! [`std::iter`]: crate::iter
//! [`std::marker`]: crate::marker
//! [`std::mem`]: crate::mem
//! [`std::ops`]: crate::ops
//! [`std::option`]: crate::option
//! [`std::prelude::v1`]: v1
//! [`std::result`]: crate::result
//! [`std::slice`]: crate::slice
//! [`std::string`]: crate::string
//! [`std::vec`]: mod@crate::vec
//! [`to_owned`]: crate::borrow::ToOwned::to_owned
//! [book-closures]: ../../book/ch13-01-closures.html
//! [book-dtor]: ../../book/ch15-03-drop.html
//! [book-enums]: ../../book/ch06-01-defining-an-enum.html
//! [book-iter]: ../../book/ch13-02-iterators.html

#![stable(feature = "rust1", since = "1.0.0")]

pub mod v1;
