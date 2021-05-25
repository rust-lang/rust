//! # The Rust Prelude
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
//! * <code>[std::marker]::{[Copy], [Send], [Sized], [Sync], [Unpin]}</code>,
//!   marker traits that indicate fundamental properties of types.
//! * <code>[std::ops]::{[Drop], [Fn], [FnMut], [FnOnce]}</code>, various
//!   operations for both destructors and overloading `()`.
//! * <code>[std::mem]::[drop][mem::drop]</code>, a convenience function for explicitly
//!   dropping a value.
//! * <code>[std::boxed]::[Box]</code>, a way to allocate values on the heap.
//! * <code>[std::borrow]::[ToOwned]</code>, the conversion trait that defines
//!   [`to_owned`], the generic method for creating an owned type from a
//!   borrowed type.
//! * <code>[std::clone]::[Clone]</code>, the ubiquitous trait that defines
//!   [`clone`][Clone::clone], the method for producing a copy of a value.
//! * <code>[std::cmp]::{[PartialEq], [PartialOrd], [Eq], [Ord]}</code>, the
//!   comparison traits, which implement the comparison operators and are often
//!   seen in trait bounds.
//! * <code>[std::convert]::{[AsRef], [AsMut], [Into], [From]}</code>, generic
//!   conversions, used by savvy API authors to create overloaded methods.
//! * <code>[std::default]::[Default]</code>, types that have default values.
//! * <code>[std::iter]::{[Iterator], [Extend], [IntoIterator], [DoubleEndedIterator], [ExactSizeIterator]}</code>,
//!   iterators of various
//!   kinds.
//! * <code>[std::option]::[Option]::{[self][Option], [Some], [None]}</code>, a
//!   type which expresses the presence or absence of a value. This type is so
//!   commonly used, its variants are also exported.
//! * <code>[std::result]::[Result]::{[self][Result], [Ok], [Err]}</code>, a type
//!   for functions that may succeed or fail. Like [`Option`], its variants are
//!   exported as well.
//! * <code>[std::string]::{[String], [ToString]}</code>, heap-allocated strings.
//! * <code>[std::vec]::[Vec]</code>, a growable, heap-allocated vector.
//!
//! [mem::drop]: crate::mem::drop
//! [std::borrow]: crate::borrow
//! [std::boxed]: crate::boxed
//! [std::clone]: crate::clone
//! [std::cmp]: crate::cmp
//! [std::convert]: crate::convert
//! [std::default]: crate::default
//! [std::iter]: crate::iter
//! [std::marker]: crate::marker
//! [std::mem]: crate::mem
//! [std::ops]: crate::ops
//! [std::option]: crate::option
//! [`std::prelude::v1`]: v1
//! [std::result]: crate::result
//! [std::slice]: crate::slice
//! [std::string]: crate::string
//! [std::vec]: mod@crate::vec
//! [`to_owned`]: crate::borrow::ToOwned::to_owned
//! [book-closures]: ../../book/ch13-01-closures.html
//! [book-dtor]: ../../book/ch15-03-drop.html
//! [book-enums]: ../../book/ch06-01-defining-an-enum.html
//! [book-iter]: ../../book/ch13-02-iterators.html

#![stable(feature = "rust1", since = "1.0.0")]

pub mod v1;

/// The 2015 version of the prelude of The Rust Standard Library.
///
/// See the [module-level documentation](self) for more.
#[unstable(feature = "prelude_2015", issue = "85684")]
pub mod rust_2015 {
    #[unstable(feature = "prelude_2015", issue = "85684")]
    #[doc(no_inline)]
    pub use super::v1::*;
}

/// The 2018 version of the prelude of The Rust Standard Library.
///
/// See the [module-level documentation](self) for more.
#[unstable(feature = "prelude_2018", issue = "85684")]
pub mod rust_2018 {
    #[unstable(feature = "prelude_2018", issue = "85684")]
    #[doc(no_inline)]
    pub use super::v1::*;
}

/// The 2021 version of the prelude of The Rust Standard Library.
///
/// See the [module-level documentation](self) for more.
#[unstable(feature = "prelude_2021", issue = "85684")]
pub mod rust_2021 {
    #[unstable(feature = "prelude_2021", issue = "85684")]
    #[doc(no_inline)]
    pub use super::v1::*;

    #[unstable(feature = "prelude_2021", issue = "85684")]
    #[doc(no_inline)]
    pub use core::prelude::rust_2021::*;
}
