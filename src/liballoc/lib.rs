// Copyright 2014-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # The Rust core allocation and collections library
//!
//! This library provides smart pointers and collections for managing
//! heap-allocated values.
//!
//! This library, like libcore, is not intended for general usage, but rather as
//! a building block of other libraries. The types and interfaces in this
//! library are re-exported through the [standard library](../std/index.html),
//! and should not be used through this library.
//!
//! ## Boxed values
//!
//! The [`Box`](boxed/index.html) type is a smart pointer type. There can
//! only be one owner of a `Box`, and the owner can decide to mutate the
//! contents, which live on the heap.
//!
//! This type can be sent among threads efficiently as the size of a `Box` value
//! is the same as that of a pointer. Tree-like data structures are often built
//! with boxes because each node often has only one owner, the parent.
//!
//! ## Reference counted pointers
//!
//! The [`Rc`](rc/index.html) type is a non-threadsafe reference-counted pointer
//! type intended for sharing memory within a thread. An `Rc` pointer wraps a
//! type, `T`, and only allows access to `&T`, a shared reference.
//!
//! This type is useful when inherited mutability (such as using `Box`) is too
//! constraining for an application, and is often paired with the `Cell` or
//! `RefCell` types in order to allow mutation.
//!
//! ## Atomically reference counted pointers
//!
//! The [`Arc`](arc/index.html) type is the threadsafe equivalent of the `Rc`
//! type. It provides all the same functionality of `Rc`, except it requires
//! that the contained type `T` is shareable. Additionally, `Arc<T>` is itself
//! sendable while `Rc<T>` is not.
//!
//! This type allows for shared access to the contained data, and is often
//! paired with synchronization primitives such as mutexes to allow mutation of
//! shared resources.
//!
//! ## Collections
//!
//! Implementations of the most common general purpose data structures are
//! defined in this library. They are re-exported through the
//! [standard collections library](../std/collections/index.html).
//!
//! ## Heap interfaces
//!
//! The [`heap`](heap/index.html) module defines the low-level interface to the
//! default global allocator. It is not compatible with the libc allocator API.

#![allow(unused_attributes)]
#![unstable(feature = "alloc",
            reason = "this library is unlikely to be stabilized in its current \
                      form or name",
            issue = "27783")]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/",
       test(no_crate_inject, attr(allow(unused_variables), deny(warnings))))]
#![no_std]
#![needs_allocator]
#![deny(warnings)]
#![deny(missing_debug_implementations)]

#![cfg_attr(test, allow(deprecated))] // rand
#![cfg_attr(test, feature(placement_in))]
#![cfg_attr(not(test), feature(core_float))]
#![cfg_attr(not(test), feature(exact_size_is_empty))]
#![cfg_attr(not(test), feature(slice_rotate))]
#![cfg_attr(not(test), feature(generator_trait))]
#![cfg_attr(test, feature(rand, test))]
#![feature(allow_internal_unstable)]
#![feature(ascii_ctype)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(cfg_target_has_atomic)]
#![feature(coerce_unsized)]
#![feature(const_fn)]
#![feature(core_intrinsics)]
#![feature(custom_attribute)]
#![feature(dropck_eyepatch)]
#![feature(exact_size_is_empty)]
#![feature(fmt_internals)]
#![feature(from_ref)]
#![feature(fundamental)]
#![feature(fused)]
#![feature(generic_param_attrs)]
#![feature(i128_type)]
#![feature(inclusive_range)]
#![feature(iter_rfold)]
#![feature(lang_items)]
#![feature(needs_allocator)]
#![feature(nonzero)]
#![feature(offset_to)]
#![feature(optin_builtin_traits)]
#![feature(pattern)]
#![feature(placement_in_syntax)]
#![feature(placement_new_protocol)]
#![feature(rustc_attrs)]
#![feature(shared)]
#![feature(slice_get_slice)]
#![feature(slice_patterns)]
#![feature(slice_rsplit)]
#![feature(specialization)]
#![feature(staged_api)]
#![feature(str_internals)]
#![feature(trusted_len)]
#![feature(unboxed_closures)]
#![feature(unicode)]
#![feature(unique)]
#![feature(unsize)]
#![feature(allocator_internals)]
#![feature(on_unimplemented)]
#![feature(exact_chunks)]

#![cfg_attr(not(test), feature(fused, fn_traits, placement_new_protocol, swap_with_slice, i128))]
#![cfg_attr(test, feature(test, box_heap))]

// Allow testing this library

#[cfg(test)]
#[macro_use]
extern crate std;
#[cfg(test)]
extern crate test;
#[cfg(test)]
extern crate rand;

extern crate std_unicode;

// Module with internal macros used by other modules (needs to be included before other modules).
#[macro_use]
mod macros;

// Allocator trait and helper struct definitions

pub mod allocator;

// Heaps provided for low-level allocation strategies

pub mod heap;

// Primitive types using the heaps above

// Need to conditionally define the mod from `boxed.rs` to avoid
// duplicating the lang-items when building in test cfg; but also need
// to allow code to have `use boxed::HEAP;`
// and `use boxed::Box;` declarations.
#[cfg(not(test))]
pub mod boxed;
#[cfg(test)]
mod boxed {
    pub use std::boxed::{Box, IntermediateBox, HEAP};
}
#[cfg(test)]
mod boxed_test;
#[cfg(target_has_atomic = "ptr")]
pub mod arc;
pub mod rc;
pub mod raw_vec;

// collections modules
pub mod binary_heap;
mod btree;
pub mod borrow;
pub mod fmt;
pub mod linked_list;
pub mod range;
pub mod slice;
pub mod str;
pub mod string;
pub mod vec;
pub mod vec_deque;

#[stable(feature = "rust1", since = "1.0.0")]
pub mod btree_map {
    //! A map based on a B-Tree.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use btree::map::*;
}

#[stable(feature = "rust1", since = "1.0.0")]
pub mod btree_set {
    //! A set based on a B-Tree.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use btree::set::*;
}

#[cfg(not(test))]
mod std {
    pub use core::ops;      // RangeFull
}

/// An endpoint of a range of keys.
///
/// # Examples
///
/// `Bound`s are range endpoints:
///
/// ```
/// #![feature(collections_range)]
///
/// use std::collections::range::RangeArgument;
/// use std::collections::Bound::*;
///
/// assert_eq!((..100).start(), Unbounded);
/// assert_eq!((1..12).start(), Included(&1));
/// assert_eq!((1..12).end(), Excluded(&12));
/// ```
///
/// Using a tuple of `Bound`s as an argument to [`BTreeMap::range`].
/// Note that in most cases, it's better to use range syntax (`1..5`) instead.
///
/// ```
/// use std::collections::BTreeMap;
/// use std::collections::Bound::{Excluded, Included, Unbounded};
///
/// let mut map = BTreeMap::new();
/// map.insert(3, "a");
/// map.insert(5, "b");
/// map.insert(8, "c");
///
/// for (key, value) in map.range((Excluded(3), Included(8))) {
///     println!("{}: {}", key, value);
/// }
///
/// assert_eq!(Some((&3, &"a")), map.range((Unbounded, Included(5))).next());
/// ```
///
/// [`BTreeMap::range`]: btree_map/struct.BTreeMap.html#method.range
#[stable(feature = "collections_bound", since = "1.17.0")]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Bound<T> {
    /// An inclusive bound.
    #[stable(feature = "collections_bound", since = "1.17.0")]
    Included(#[stable(feature = "collections_bound", since = "1.17.0")] T),
    /// An exclusive bound.
    #[stable(feature = "collections_bound", since = "1.17.0")]
    Excluded(#[stable(feature = "collections_bound", since = "1.17.0")] T),
    /// An infinite endpoint. Indicates that there is no bound in this direction.
    #[stable(feature = "collections_bound", since = "1.17.0")]
    Unbounded,
}

/// An intermediate trait for specialization of `Extend`.
#[doc(hidden)]
trait SpecExtend<I: IntoIterator> {
    /// Extends `self` with the contents of the given iterator.
    fn spec_extend(&mut self, iter: I);
}

#[doc(no_inline)]
pub use binary_heap::BinaryHeap;
#[doc(no_inline)]
pub use btree_map::BTreeMap;
#[doc(no_inline)]
pub use btree_set::BTreeSet;
#[doc(no_inline)]
pub use linked_list::LinkedList;
#[doc(no_inline)]
pub use vec_deque::VecDeque;
#[doc(no_inline)]
pub use string::String;
#[doc(no_inline)]
pub use vec::Vec;
