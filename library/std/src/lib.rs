//! # The Rust Standard Library
//!
//! The Rust Standard Library is the foundation of portable Rust software, a
//! set of minimal and battle-tested shared abstractions for the [broader Rust
//! ecosystem][crates.io]. It offers core types, like [`Vec<T>`] and
//! [`Option<T>`], library-defined [operations on language
//! primitives](#primitives), [standard macros](#macros), [I/O] and
//! [multithreading], among [many other things][other].
//!
//! `std` is available to all Rust crates by default. Therefore, the
//! standard library can be accessed in [`use`] statements through the path
//! `std`, as in [`use std::env`].
//!
//! # How to read this documentation
//!
//! If you already know the name of what you are looking for, the fastest way to
//! find it is to use the <a href="#" onclick="window.searchState.focus();">search
//! bar</a> at the top of the page.
//!
//! Otherwise, you may want to jump to one of these useful sections:
//!
//! * [`std::*` modules](#modules)
//! * [Primitive types](#primitives)
//! * [Standard macros](#macros)
//! * [The Rust Prelude]
//!
//! If this is your first time, the documentation for the standard library is
//! written to be casually perused. Clicking on interesting things should
//! generally lead you to interesting places. Still, there are important bits
//! you don't want to miss, so read on for a tour of the standard library and
//! its documentation!
//!
//! Once you are familiar with the contents of the standard library you may
//! begin to find the verbosity of the prose distracting. At this stage in your
//! development you may want to press the `[-]` button near the top of the
//! page to collapse it into a more skimmable view.
//!
//! While you are looking at that `[-]` button also notice the `source`
//! link. Rust's API documentation comes with the source code and you are
//! encouraged to read it. The standard library source is generally high
//! quality and a peek behind the curtains is often enlightening.
//!
//! # What is in the standard library documentation?
//!
//! First of all, The Rust Standard Library is divided into a number of focused
//! modules, [all listed further down this page](#modules). These modules are
//! the bedrock upon which all of Rust is forged, and they have mighty names
//! like [`std::slice`] and [`std::cmp`]. Modules' documentation typically
//! includes an overview of the module along with examples, and are a smart
//! place to start familiarizing yourself with the library.
//!
//! Second, implicit methods on [primitive types] are documented here. This can
//! be a source of confusion for two reasons:
//!
//! 1. While primitives are implemented by the compiler, the standard library
//!    implements methods directly on the primitive types (and it is the only
//!    library that does so), which are [documented in the section on
//!    primitives](#primitives).
//! 2. The standard library exports many modules *with the same name as
//!    primitive types*. These define additional items related to the primitive
//!    type, but not the all-important methods.
//!
//! So for example there is a [page for the primitive type
//! `i32`](primitive::i32) that lists all the methods that can be called on
//! 32-bit integers (very useful), and there is a [page for the module
//! `std::i32`] that documents the constant values [`MIN`] and [`MAX`] (rarely
//! useful).
//!
//! Note the documentation for the primitives [`str`] and [`[T]`][prim@slice] (also
//! called 'slice'). Many method calls on [`String`] and [`Vec<T>`] are actually
//! calls to methods on [`str`] and [`[T]`][prim@slice] respectively, via [deref
//! coercions][deref-coercions].
//!
//! Third, the standard library defines [The Rust Prelude], a small collection
//! of items - mostly traits - that are imported into every module of every
//! crate. The traits in the prelude are pervasive, making the prelude
//! documentation a good entry point to learning about the library.
//!
//! And finally, the standard library exports a number of standard macros, and
//! [lists them on this page](#macros) (technically, not all of the standard
//! macros are defined by the standard library - some are defined by the
//! compiler - but they are documented here the same). Like the prelude, the
//! standard macros are imported by default into all crates.
//!
//! # Contributing changes to the documentation
//!
//! Check out the rust contribution guidelines [here](
//! https://rustc-dev-guide.rust-lang.org/contributing.html#writing-documentation).
//! The source for this documentation can be found on
//! [GitHub](https://github.com/rust-lang/rust).
//! To contribute changes, make sure you read the guidelines first, then submit
//! pull-requests for your suggested changes.
//!
//! Contributions are appreciated! If you see a part of the docs that can be
//! improved, submit a PR, or chat with us first on [Discord][rust-discord]
//! #docs.
//!
//! # A Tour of The Rust Standard Library
//!
//! The rest of this crate documentation is dedicated to pointing out notable
//! features of The Rust Standard Library.
//!
//! ## Containers and collections
//!
//! The [`option`] and [`result`] modules define optional and error-handling
//! types, [`Option<T>`] and [`Result<T, E>`]. The [`iter`] module defines
//! Rust's iterator trait, [`Iterator`], which works with the [`for`] loop to
//! access collections.
//!
//! The standard library exposes three common ways to deal with contiguous
//! regions of memory:
//!
//! * [`Vec<T>`] - A heap-allocated *vector* that is resizable at runtime.
//! * [`[T; N]`][prim@array] - An inline *array* with a fixed size at compile time.
//! * [`[T]`][prim@slice] - A dynamically sized *slice* into any other kind of contiguous
//!   storage, whether heap-allocated or not.
//!
//! Slices can only be handled through some kind of *pointer*, and as such come
//! in many flavors such as:
//!
//! * `&[T]` - *shared slice*
//! * `&mut [T]` - *mutable slice*
//! * [`Box<[T]>`][owned slice] - *owned slice*
//!
//! [`str`], a UTF-8 string slice, is a primitive type, and the standard library
//! defines many methods for it. Rust [`str`]s are typically accessed as
//! immutable references: `&str`. Use the owned [`String`] for building and
//! mutating strings.
//!
//! For converting to strings use the [`format!`] macro, and for converting from
//! strings use the [`FromStr`] trait.
//!
//! Data may be shared by placing it in a reference-counted box or the [`Rc`]
//! type, and if further contained in a [`Cell`] or [`RefCell`], may be mutated
//! as well as shared. Likewise, in a concurrent setting it is common to pair an
//! atomically-reference-counted box, [`Arc`], with a [`Mutex`] to get the same
//! effect.
//!
//! The [`collections`] module defines maps, sets, linked lists and other
//! typical collection types, including the common [`HashMap<K, V>`].
//!
//! ## Platform abstractions and I/O
//!
//! Besides basic data types, the standard library is largely concerned with
//! abstracting over differences in common platforms, most notably Windows and
//! Unix derivatives.
//!
//! Common types of I/O, including [files], [TCP], and [UDP], are defined in
//! the [`io`], [`fs`], and [`net`] modules.
//!
//! The [`thread`] module contains Rust's threading abstractions. [`sync`]
//! contains further primitive shared memory types, including [`atomic`] and
//! [`mpsc`], which contains the channel types for message passing.
//!
//! [I/O]: io
//! [`MIN`]: i32::MIN
//! [`MAX`]: i32::MAX
//! [page for the module `std::i32`]: crate::i32
//! [TCP]: net::TcpStream
//! [The Rust Prelude]: prelude
//! [UDP]: net::UdpSocket
//! [`Arc`]: sync::Arc
//! [owned slice]: boxed
//! [`Cell`]: cell::Cell
//! [`FromStr`]: str::FromStr
//! [`HashMap<K, V>`]: collections::HashMap
//! [`Mutex`]: sync::Mutex
//! [`Option<T>`]: option::Option
//! [`Rc`]: rc::Rc
//! [`RefCell`]: cell::RefCell
//! [`Result<T, E>`]: result::Result
//! [`Vec<T>`]: vec::Vec
//! [`atomic`]: sync::atomic
//! [`for`]: ../book/ch03-05-control-flow.html#looping-through-a-collection-with-for
//! [`str`]: prim@str
//! [`mpsc`]: sync::mpsc
//! [`std::cmp`]: cmp
//! [`std::slice`]: mod@slice
//! [`use std::env`]: env/index.html
//! [`use`]: ../book/ch07-02-defining-modules-to-control-scope-and-privacy.html
//! [crates.io]: https://crates.io
//! [deref-coercions]: ../book/ch15-02-deref.html#implicit-deref-coercions-with-functions-and-methods
//! [files]: fs::File
//! [multithreading]: thread
//! [other]: #what-is-in-the-standard-library-documentation
//! [primitive types]: ../book/ch03-02-data-types.html
//! [rust-discord]: https://discord.gg/rust-lang
//! [array]: prim@array
//! [slice]: prim@slice

// To run std tests without x.py without ending up with two copies of std, Miri needs to be
// able to "empty" this crate. See <https://github.com/rust-lang/miri-test-libstd/issues/4>.
// rustc itself never sets the feature, so this line has no affect there.
#![cfg(any(not(feature = "miri-test-libstd"), test, doctest))]
// miri-test-libstd also prefers to make std use the sysroot versions of the dependencies.
#![cfg_attr(feature = "miri-test-libstd", feature(rustc_private))]
//
#![cfg_attr(not(feature = "restricted-std"), stable(feature = "rust1", since = "1.0.0"))]
#![cfg_attr(feature = "restricted-std", unstable(feature = "restricted_std", issue = "none"))]
#![doc(
    html_playground_url = "https://play.rust-lang.org/",
    issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/",
    test(no_crate_inject, attr(deny(warnings))),
    test(attr(allow(dead_code, deprecated, unused_variables, unused_mut)))
)]
#![doc(cfg_hide(
    not(test),
    not(any(test, bootstrap)),
    no_global_oom_handling,
    not(no_global_oom_handling)
))]
// Don't link to std. We are std.
#![no_std]
// Tell the compiler to link to either panic_abort or panic_unwind
#![needs_panic_runtime]
//
// Lints:
#![warn(deprecated_in_future)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![allow(explicit_outlives_requirements)]
#![allow(unused_lifetimes)]
#![deny(rustc::existing_doc_keyword)]
#![deny(fuzzy_provenance_casts)]
// Ensure that std can be linked against panic_abort despite compiled with `-C panic=unwind`
#![deny(ffi_unwind_calls)]
// std may use features in a platform-specific way
#![allow(unused_features)]
//
// Features:
#![cfg_attr(test, feature(internal_output_capture, print_internals, update_panic_count, rt))]
#![cfg_attr(
    all(target_vendor = "fortanix", target_env = "sgx"),
    feature(slice_index_methods, coerce_unsized, sgx_platform)
)]
#![cfg_attr(windows, feature(round_char_boundary))]
//
// Language features:
// tidy-alphabetical-start
#![feature(alloc_error_handler)]
#![feature(allocator_internals)]
#![feature(allow_internal_unsafe)]
#![feature(allow_internal_unstable)]
#![feature(c_unwind)]
#![feature(cfg_target_thread_local)]
#![feature(concat_idents)]
#![feature(const_mut_refs)]
#![feature(const_trait_impl)]
#![feature(decl_macro)]
#![feature(deprecated_suggestion)]
#![feature(doc_cfg)]
#![feature(doc_cfg_hide)]
#![feature(doc_masked)]
#![feature(doc_notable_trait)]
#![feature(dropck_eyepatch)]
#![feature(exhaustive_patterns)]
#![feature(if_let_guard)]
#![feature(intra_doc_pointers)]
#![feature(lang_items)]
#![feature(let_chains)]
#![feature(link_cfg)]
#![feature(linkage)]
#![feature(min_specialization)]
#![feature(must_not_suspend)]
#![feature(needs_panic_runtime)]
#![feature(negative_impls)]
#![feature(never_type)]
#![feature(platform_intrinsics)]
#![feature(prelude_import)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![feature(staged_api)]
#![feature(thread_local)]
#![feature(try_blocks)]
#![feature(utf8_chunks)]
// tidy-alphabetical-end
//
// Library features (core):
// tidy-alphabetical-start
#![feature(char_internals)]
#![feature(core_intrinsics)]
#![feature(duration_constants)]
#![feature(error_generic_member_access)]
#![feature(error_in_core)]
#![feature(error_iter)]
#![feature(exact_size_is_empty)]
#![feature(exclusive_wrapper)]
#![feature(extend_one)]
#![feature(float_minimum_maximum)]
#![feature(float_next_up_down)]
#![feature(hasher_prefixfree_extras)]
#![feature(hashmap_internals)]
#![feature(int_roundings)]
#![feature(ip)]
#![feature(ip_in_core)]
#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_write_slice)]
#![feature(panic_can_unwind)]
#![feature(panic_info_message)]
#![feature(panic_internals)]
#![feature(pointer_byte_offsets)]
#![feature(pointer_is_aligned)]
#![feature(portable_simd)]
#![feature(prelude_2024)]
#![feature(provide_any)]
#![feature(ptr_as_uninit)]
#![feature(raw_os_nonzero)]
#![feature(round_ties_even)]
#![feature(slice_internals)]
#![feature(slice_ptr_get)]
#![feature(std_internals)]
#![feature(str_internals)]
#![feature(strict_provenance)]
// tidy-alphabetical-end
//
// Library features (alloc):
// tidy-alphabetical-start
#![feature(alloc_layout_extra)]
#![feature(allocator_api)]
#![feature(get_mut_unchecked)]
#![feature(map_try_insert)]
#![feature(new_uninit)]
#![feature(slice_concat_trait)]
#![feature(thin_box)]
#![feature(try_reserve_kind)]
#![feature(vec_into_raw_parts)]
// tidy-alphabetical-end
//
// Library features (unwind):
// tidy-alphabetical-start
#![feature(panic_unwind)]
// tidy-alphabetical-end
//
// Only for re-exporting:
// tidy-alphabetical-start
#![feature(assert_matches)]
#![feature(async_iterator)]
#![feature(c_variadic)]
#![feature(cfg_accessible)]
#![feature(cfg_eval)]
#![feature(concat_bytes)]
#![feature(const_format_args)]
#![feature(core_panic)]
#![feature(custom_test_frameworks)]
#![feature(edition_panic)]
#![feature(format_args_nl)]
#![feature(get_many_mut)]
#![feature(lazy_cell)]
#![feature(log_syntax)]
#![feature(saturating_int_impl)]
#![feature(stdsimd)]
#![feature(test)]
#![feature(trace_macros)]
// tidy-alphabetical-end
//
// Only used in tests/benchmarks:
//
// Only for const-ness:
// tidy-alphabetical-start
#![feature(const_collections_with_hasher)]
#![feature(const_hash)]
#![feature(const_io_structs)]
#![feature(const_ip)]
#![feature(const_ipv4)]
#![feature(const_ipv6)]
#![feature(const_maybe_uninit_uninit_array)]
#![feature(const_waker)]
#![feature(thread_local_internals)]
// tidy-alphabetical-end
//
#![default_lib_allocator]

// Explicitly import the prelude. The compiler uses this same unstable attribute
// to import the prelude implicitly when building crates that depend on std.
#[prelude_import]
#[allow(unused)]
use prelude::rust_2021::*;

// Access to Bencher, etc.
#[cfg(test)]
extern crate test;

#[allow(unused_imports)] // macros from `alloc` are not used on all platforms
#[macro_use]
extern crate alloc as alloc_crate;
#[doc(masked)]
#[allow(unused_extern_crates)]
extern crate libc;

// We always need an unwinder currently for backtraces
#[doc(masked)]
#[allow(unused_extern_crates)]
extern crate unwind;

#[doc(masked)]
#[allow(unused_extern_crates)]
#[cfg(feature = "miniz_oxide")]
extern crate miniz_oxide;

// During testing, this crate is not actually the "real" std library, but rather
// it links to the real std library, which was compiled from this same source
// code. So any lang items std defines are conditionally excluded (or else they
// would generate duplicate lang item errors), and any globals it defines are
// _not_ the globals used by "real" std. So this import, defined only during
// testing gives test-std access to real-std lang items and globals. See #2912
#[cfg(test)]
extern crate std as realstd;

// The standard macros that are not built-in to the compiler.
#[macro_use]
mod macros;

// The runtime entry point and a few unstable public functions used by the
// compiler
#[macro_use]
pub mod rt;

// The Rust prelude
pub mod prelude;

// Public module declarations and re-exports
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::borrow;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::boxed;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::fmt;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::format;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::rc;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::slice;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::str;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::string;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::vec;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::any;
#[stable(feature = "core_array", since = "1.36.0")]
pub use core::array;
#[unstable(feature = "async_iterator", issue = "79024")]
pub use core::async_iter;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::cell;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::char;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::clone;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::cmp;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::convert;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::default;
#[stable(feature = "futures_api", since = "1.36.0")]
pub use core::future;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::hash;
#[stable(feature = "core_hint", since = "1.27.0")]
pub use core::hint;
#[stable(feature = "i128", since = "1.26.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::i128;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::i16;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::i32;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::i64;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::i8;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::intrinsics;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::isize;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::iter;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::marker;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::mem;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::ops;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::option;
#[stable(feature = "pin", since = "1.33.0")]
pub use core::pin;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::ptr;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::result;
#[stable(feature = "i128", since = "1.26.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::u128;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::u16;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::u32;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::u64;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::u8;
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::usize;

pub mod f32;
pub mod f64;

#[macro_use]
pub mod thread;
pub mod ascii;
pub mod backtrace;
pub mod collections;
pub mod env;
pub mod error;
pub mod ffi;
pub mod fs;
pub mod io;
pub mod net;
pub mod num;
pub mod os;
pub mod panic;
pub mod path;
pub mod process;
pub mod sync;
pub mod time;

// Pull in `std_float` crate  into std. The contents of
// `std_float` are in a different repository: rust-lang/portable-simd.
#[path = "../../portable-simd/crates/std_float/src/lib.rs"]
#[allow(missing_debug_implementations, dead_code, unsafe_op_in_unsafe_fn, unused_unsafe)]
#[allow(rustdoc::bare_urls)]
#[unstable(feature = "portable_simd", issue = "86656")]
mod std_float;

#[doc = include_str!("../../portable-simd/crates/core_simd/src/core_simd_docs.md")]
#[unstable(feature = "portable_simd", issue = "86656")]
pub mod simd {
    #[doc(inline)]
    pub use crate::std_float::StdFloat;
    #[doc(inline)]
    pub use core::simd::*;
}

#[stable(feature = "futures_api", since = "1.36.0")]
pub mod task {
    //! Types and Traits for working with asynchronous tasks.

    #[doc(inline)]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub use core::task::*;

    #[doc(inline)]
    #[stable(feature = "wake_trait", since = "1.51.0")]
    pub use alloc::task::*;
}

#[doc = include_str!("../../stdarch/crates/core_arch/src/core_arch_docs.md")]
#[stable(feature = "simd_arch", since = "1.27.0")]
pub mod arch {
    #[stable(feature = "simd_arch", since = "1.27.0")]
    // The `no_inline`-attribute is required to make the documentation of all
    // targets available.
    // See https://github.com/rust-lang/rust/pull/57808#issuecomment-457390549 for
    // more information.
    #[doc(no_inline)] // Note (#82861): required for correct documentation
    pub use core::arch::*;

    #[stable(feature = "simd_aarch64", since = "1.60.0")]
    pub use std_detect::is_aarch64_feature_detected;
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub use std_detect::is_x86_feature_detected;
    #[unstable(feature = "stdsimd", issue = "48556")]
    pub use std_detect::{
        is_arm_feature_detected, is_mips64_feature_detected, is_mips_feature_detected,
        is_powerpc64_feature_detected, is_powerpc_feature_detected, is_riscv_feature_detected,
    };
}

// This was stabilized in the crate root so we have to keep it there.
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use std_detect::is_x86_feature_detected;

// Platform-abstraction modules
mod sys;
mod sys_common;

pub mod alloc;

// Private support modules
mod panicking;

#[unstable(feature = "ice_to_disk", issue = "none")]
pub use panicking::panic_hook_with_disk_dump;

#[path = "../../backtrace/src/lib.rs"]
#[allow(dead_code, unused_attributes, fuzzy_provenance_casts)]
mod backtrace_rs;

// Re-export macros defined in core.
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated, deprecated_in_future)]
pub use core::{
    assert_eq, assert_ne, debug_assert, debug_assert_eq, debug_assert_ne, matches, todo, r#try,
    unimplemented, unreachable, write, writeln,
};

// Re-export built-in macros defined through core.
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[allow(deprecated)]
pub use core::{
    assert, assert_matches, cfg, column, compile_error, concat, concat_idents, const_format_args,
    env, file, format_args, format_args_nl, include, include_bytes, include_str, line, log_syntax,
    module_path, option_env, stringify, trace_macros,
};

#[unstable(
    feature = "concat_bytes",
    issue = "87555",
    reason = "`concat_bytes` is not stable enough for use and is subject to change"
)]
pub use core::concat_bytes;

#[stable(feature = "core_primitive", since = "1.43.0")]
pub use core::primitive;

// Include a number of private modules that exist solely to provide
// the rustdoc documentation for primitive types. Using `include!`
// because rustdoc only looks for these modules at the crate level.
include!("primitive_docs.rs");

// Include a number of private modules that exist solely to provide
// the rustdoc documentation for the existing keywords. Using `include!`
// because rustdoc only looks for these modules at the crate level.
include!("keyword_docs.rs");

// This is required to avoid an unstable error when `restricted-std` is not
// enabled. The use of #![feature(restricted_std)] in rustc-std-workspace-std
// is unconditional, so the unstable feature needs to be defined somewhere.
#[unstable(feature = "restricted_std", issue = "none")]
mod __restricted_std_workaround {}

mod sealed {
    /// This trait being unreachable from outside the crate
    /// prevents outside implementations of our extension traits.
    /// This allows adding more trait methods in the future.
    #[unstable(feature = "sealed", issue = "none")]
    pub trait Sealed {}
}

#[cfg(test)]
#[allow(dead_code)] // Not used in all configurations.
pub(crate) mod test_helpers {
    /// Test-only replacement for `rand::thread_rng()`, which is unusable for
    /// us, as we want to allow running stdlib tests on tier-3 targets which may
    /// not have `getrandom` support.
    ///
    /// Does a bit of a song and dance to ensure that the seed is different on
    /// each call (as some tests sadly rely on this), but doesn't try that hard.
    ///
    /// This is duplicated in the `core`, `alloc` test suites (as well as
    /// `std`'s integration tests), but figuring out a mechanism to share these
    /// seems far more painful than copy-pasting a 7 line function a couple
    /// times, given that even under a perma-unstable feature, I don't think we
    /// want to expose types from `rand` from `std`.
    #[track_caller]
    pub(crate) fn test_rng() -> rand_xorshift::XorShiftRng {
        use core::hash::{BuildHasher, Hash, Hasher};
        let mut hasher = crate::collections::hash_map::RandomState::new().build_hasher();
        core::panic::Location::caller().hash(&mut hasher);
        let hc64 = hasher.finish();
        let seed_vec = hc64.to_le_bytes().into_iter().chain(0u8..8).collect::<Vec<u8>>();
        let seed: [u8; 16] = seed_vec.as_slice().try_into().unwrap();
        rand::SeedableRng::from_seed(seed)
    }
}
