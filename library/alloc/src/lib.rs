//! # The Rust core allocation and collections library
//!
//! This library provides smart pointers and collections for managing
//! heap-allocated values.
//!
//! This library, like libcore, normally doesn’t need to be used directly
//! since its contents are re-exported in the [`std` crate](../std/index.html).
//! Crates that use the `#![no_std]` attribute however will typically
//! not depend on `std`, so they’d use this crate instead.
//!
//! ## Boxed values
//!
//! The [`Box`] type is a smart pointer type. There can only be one owner of a
//! [`Box`], and the owner can decide to mutate the contents, which live on the
//! heap.
//!
//! This type can be sent among threads efficiently as the size of a `Box` value
//! is the same as that of a pointer. Tree-like data structures are often built
//! with boxes because each node often has only one owner, the parent.
//!
//! ## Reference counted pointers
//!
//! The [`Rc`] type is a non-threadsafe reference-counted pointer type intended
//! for sharing memory within a thread. An [`Rc`] pointer wraps a type, `T`, and
//! only allows access to `&T`, a shared reference.
//!
//! This type is useful when inherited mutability (such as using [`Box`]) is too
//! constraining for an application, and is often paired with the [`Cell`] or
//! [`RefCell`] types in order to allow mutation.
//!
//! ## Atomically reference counted pointers
//!
//! The [`Arc`] type is the threadsafe equivalent of the [`Rc`] type. It
//! provides all the same functionality of [`Rc`], except it requires that the
//! contained type `T` is shareable. Additionally, [`Arc<T>`][`Arc`] is itself
//! sendable while [`Rc<T>`][`Rc`] is not.
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
//! The [`alloc`](alloc/index.html) module defines the low-level interface to the
//! default global allocator. It is not compatible with the libc allocator API.
//!
//! [`Arc`]: sync
//! [`Box`]: boxed
//! [`Cell`]: core::cell
//! [`Rc`]: rc
//! [`RefCell`]: core::cell

// To run liballoc tests without x.py without ending up with two copies of liballoc, Miri needs to be
// able to "empty" this crate. See <https://github.com/rust-lang/miri-test-libstd/issues/4>.
// rustc itself never sets the feature, so this line has no affect there.
#![cfg(any(not(feature = "miri-test-libstd"), test, doctest))]
#![allow(unused_attributes)]
#![stable(feature = "alloc", since = "1.36.0")]
#![doc(
    html_playground_url = "https://play.rust-lang.org/",
    issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/",
    test(no_crate_inject, attr(allow(unused_variables), deny(warnings)))
)]
#![no_std]
#![needs_allocator]
#![warn(deprecated_in_future)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![allow(explicit_outlives_requirements)]
#![deny(unsafe_op_in_unsafe_fn)]
#![feature(rustc_allow_const_fn_unstable)]
#![cfg_attr(not(test), feature(generator_trait))]
#![cfg_attr(test, feature(test))]
#![cfg_attr(test, feature(new_uninit))]
#![feature(allocator_api)]
#![feature(array_chunks)]
#![feature(array_methods)]
#![feature(array_windows)]
#![feature(allow_internal_unstable)]
#![feature(arbitrary_self_types)]
#![feature(async_stream)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(cfg_sanitize)]
#![feature(cfg_target_has_atomic)]
#![feature(coerce_unsized)]
#![cfg_attr(not(no_global_oom_handling), feature(const_btree_new))]
#![feature(const_fn_trait_bound)]
#![feature(cow_is_borrowed)]
#![feature(const_cow_is_borrowed)]
#![feature(const_trait_impl)]
#![feature(destructuring_assignment)]
#![feature(dispatch_from_dyn)]
#![feature(core_intrinsics)]
#![feature(dropck_eyepatch)]
#![feature(exact_size_is_empty)]
#![feature(exclusive_range_pattern)]
#![feature(extend_one)]
#![feature(fmt_internals)]
#![feature(fn_traits)]
#![feature(fundamental)]
#![feature(inplace_iteration)]
// Technically, this is a bug in rustdoc: rustdoc sees the documentation on `#[lang = slice_alloc]`
// blocks is for `&[T]`, which also has documentation using this feature in `core`, and gets mad
// that the feature-gate isn't enabled. Ideally, it wouldn't check for the feature gate for docs
// from other crates, but since this can only appear for lang items, it doesn't seem worth fixing.
#![feature(intra_doc_pointers)]
#![feature(iter_zip)]
#![feature(lang_items)]
#![feature(layout_for_ptr)]
#![feature(negative_impls)]
#![feature(never_type)]
#![feature(nll)]
#![feature(nonnull_slice_from_raw_parts)]
#![feature(auto_traits)]
#![feature(option_result_unwrap_unchecked)]
#![feature(pattern)]
#![feature(ptr_internals)]
#![feature(rustc_attrs)]
#![feature(receiver_trait)]
#![feature(min_specialization)]
#![feature(set_ptr_value)]
#![feature(slice_ptr_get)]
#![feature(slice_ptr_len)]
#![feature(slice_range)]
#![feature(staged_api)]
#![feature(str_internals)]
#![feature(trusted_len)]
#![feature(unboxed_closures)]
#![feature(unicode_internals)]
#![feature(unsize)]
#![feature(unsized_fn_params)]
#![feature(allocator_internals)]
#![feature(slice_partition_dedup)]
#![feature(maybe_uninit_extra, maybe_uninit_slice, maybe_uninit_uninit_array)]
#![feature(alloc_layout_extra)]
#![feature(trusted_random_access)]
#![feature(try_trait_v2)]
#![feature(associated_type_bounds)]
#![feature(slice_group_by)]
#![feature(decl_macro)]
// Allow testing this library

#[cfg(test)]
#[macro_use]
extern crate std;
#[cfg(test)]
extern crate test;

// Module with internal macros used by other modules (needs to be included before other modules).
#[macro_use]
mod macros;

// Heaps provided for low-level allocation strategies

pub mod alloc;

// Primitive types using the heaps above

// Need to conditionally define the mod from `boxed.rs` to avoid
// duplicating the lang-items when building in test cfg; but also need
// to allow code to have `use boxed::Box;` declarations.
#[cfg(not(test))]
pub mod boxed;
#[cfg(test)]
mod boxed {
    pub use std::boxed::Box;
}
pub mod borrow;
pub mod collections;
pub mod fmt;
pub mod prelude;
pub mod raw_vec;
pub mod rc;
pub mod slice;
pub mod str;
pub mod string;
#[cfg(target_has_atomic = "ptr")]
pub mod sync;
#[cfg(all(not(no_global_oom_handling), target_has_atomic = "ptr"))]
pub mod task;
#[cfg(test)]
mod tests;
pub mod vec;

#[doc(hidden)]
#[unstable(feature = "liballoc_internals", issue = "none", reason = "implementation detail")]
pub mod __export {
    pub use core::format_args;
}
