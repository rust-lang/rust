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
//! find it is to use the <a href="#" onclick="focusSearchBar();">search
//! bar</a> at the top of the page.
//!
//! Otherwise, you may want to jump to one of these useful sections:
//!
//! * [`std::*` modules](#modules)
//! * [Primitive types](#primitives)
//! * [Standard macros](#macros)
//! * [The Rust Prelude](prelude/index.html)
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
//! While you are looking at that `[-]` button also notice the `[src]`
//! button. Rust's API documentation comes with the source code and you are
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
//! `i32`](primitive.i32.html) that lists all the methods that can be called on
//! 32-bit integers (very useful), and there is a [page for the module
//! `std::i32`](i32/index.html) that documents the constant values [`MIN`] and
//! [`MAX`](i32/constant.MAX.html) (rarely useful).
//!
//! Note the documentation for the primitives [`str`] and [`[T]`][slice] (also
//! called 'slice'). Many method calls on [`String`] and [`Vec<T>`] are actually
//! calls to methods on [`str`] and [`[T]`][slice] respectively, via [deref
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
//! https://github.com/rust-lang/rust/blob/master/CONTRIBUTING.md).
//! The source for this documentation can be found on [Github](https://github.com/rust-lang).
//! To contribute changes, make sure you read the guidelines first, then submit
//! pull-requests for your suggested changes.
//!
//! Contributions are appreciated! If you see a part of the docs that can be
//! improved, submit a PR, or chat with us first on irc.mozilla.org #rust-docs.
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
//! * [`[T; n]`][array] - An inline *array* with a fixed size at compile time.
//! * [`[T]`][slice] - A dynamically sized *slice* into any other kind of contiguous
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
//! Common types of I/O, including [files], [TCP], [UDP], are defined in the
//! [`io`], [`fs`], and [`net`] modules.
//!
//! The [`thread`] module contains Rust's threading abstractions. [`sync`]
//! contains further primitive shared memory types, including [`atomic`] and
//! [`mpsc`], which contains the channel types for message passing.
//!
//! [I/O]: io/index.html
//! [`MIN`]: i32/constant.MIN.html
//! [TCP]: net/struct.TcpStream.html
//! [The Rust Prelude]: prelude/index.html
//! [UDP]: net/struct.UdpSocket.html
//! [`Arc`]: sync/struct.Arc.html
//! [owned slice]: boxed/index.html
//! [`Cell`]: cell/struct.Cell.html
//! [`FromStr`]: str/trait.FromStr.html
//! [`HashMap<K, V>`]: collections/struct.HashMap.html
//! [`Iterator`]: iter/trait.Iterator.html
//! [`Mutex`]: sync/struct.Mutex.html
//! [`Option<T>`]: option/enum.Option.html
//! [`Rc`]: rc/index.html
//! [`RefCell`]: cell/struct.RefCell.html
//! [`Result<T, E>`]: result/enum.Result.html
//! [`String`]: string/struct.String.html
//! [`Vec<T>`]: vec/index.html
//! [array]: primitive.array.html
//! [slice]: primitive.slice.html
//! [`atomic`]: sync/atomic/index.html
//! [`collections`]: collections/index.html
//! [`for`]: ../book/ch03-05-control-flow.html#looping-through-a-collection-with-for
//! [`format!`]: macro.format.html
//! [`fs`]: fs/index.html
//! [`io`]: io/index.html
//! [`iter`]: iter/index.html
//! [`mpsc`]: sync/mpsc/index.html
//! [`net`]: net/index.html
//! [`option`]: option/index.html
//! [`result`]: result/index.html
//! [`std::cmp`]: cmp/index.html
//! [`std::slice`]: slice/index.html
//! [`str`]: primitive.str.html
//! [`sync`]: sync/index.html
//! [`thread`]: thread/index.html
//! [`use std::env`]: env/index.html
//! [`use`]: ../book/ch07-02-defining-modules-to-control-scope-and-privacy.html
//! [crates.io]: https://crates.io
//! [deref-coercions]: ../book/ch15-02-deref.html#implicit-deref-coercions-with-functions-and-methods
//! [files]: fs/struct.File.html
//! [multithreading]: thread/index.html
//! [other]: #what-is-in-the-standard-library-documentation
//! [primitive types]: ../book/ch03-02-data-types.html

#![stable(feature = "rust1", since = "1.0.0")]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/",
       issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/",
       test(no_crate_inject, attr(deny(warnings))),
       test(attr(allow(dead_code, deprecated, unused_variables, unused_mut))))]

// Don't link to std. We are std.
#![no_std]

//#![warn(deprecated_in_future)] // FIXME: std still has quite a few uses of `mem::uninitialized`
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![deny(intra_doc_link_resolution_failure)] // rustdoc is run without -D warnings

#![deny(rust_2018_idioms)]
#![allow(explicit_outlives_requirements)]

// Tell the compiler to link to either panic_abort or panic_unwind
#![needs_panic_runtime]

// std may use features in a platform-specific way
#![allow(unused_features)]

#![cfg_attr(test, feature(print_internals, set_stdio, test, update_panic_count))]
#![cfg_attr(all(target_vendor = "fortanix", target_env = "sgx"),
            feature(global_asm, slice_index_methods,
                    decl_macro, coerce_unsized, sgx_platform, ptr_wrapping_offset_from))]
#![cfg_attr(all(test, target_vendor = "fortanix", target_env = "sgx"),
            feature(fixed_size_array, maybe_uninit_extra))]

// std is implemented with unstable features, many of which are internal
// compiler details that will never be stable
// NB: the following list is sorted to minimize merge conflicts.
#![feature(alloc_error_handler)]
#![feature(alloc_layout_extra)]
#![feature(allocator_api)]
#![feature(allocator_internals)]
#![feature(allow_internal_unsafe)]
#![feature(allow_internal_unstable)]
#![feature(arbitrary_self_types)]
#![feature(array_error_internals)]
#![feature(asm)]
#![feature(bind_by_move_pattern_guards)]
#![feature(box_syntax)]
#![feature(c_variadic)]
#![feature(cfg_target_has_atomic)]
#![feature(cfg_target_thread_local)]
#![feature(char_error_internals)]
#![feature(checked_duration_since)]
#![feature(clamp)]
#![feature(compiler_builtins_lib)]
#![feature(concat_idents)]
#![feature(const_cstr_unchecked)]
#![feature(const_raw_ptr_deref)]
#![feature(core_intrinsics)]
#![feature(doc_alias)]
#![feature(doc_cfg)]
#![feature(doc_keyword)]
#![feature(doc_masked)]
#![feature(doc_spotlight)]
#![feature(dropck_eyepatch)]
#![feature(duration_constants)]
#![feature(exact_size_is_empty)]
#![feature(exhaustive_patterns)]
#![feature(external_doc)]
#![feature(fn_traits)]
#![feature(generator_trait)]
#![feature(hash_raw_entry)]
#![feature(hashmap_internals)]
#![feature(int_error_internals)]
#![feature(int_error_matching)]
#![feature(integer_atomics)]
#![feature(lang_items)]
#![feature(libc)]
#![feature(link_args)]
#![feature(linkage)]
#![feature(mem_take)]
#![feature(needs_panic_runtime)]
#![feature(never_type)]
#![feature(nll)]
#![feature(non_exhaustive)]
#![feature(on_unimplemented)]
#![feature(optin_builtin_traits)]
#![feature(panic_info_message)]
#![feature(panic_internals)]
#![feature(panic_unwind)]
#![feature(prelude_import)]
#![feature(ptr_internals)]
#![feature(raw)]
#![feature(renamed_spin_loop)]
#![feature(rustc_attrs)]
#![feature(rustc_const_unstable)]
#![feature(rustc_private)]
#![feature(shrink_to)]
#![feature(slice_concat_ext)]
#![feature(slice_internals)]
#![feature(slice_patterns)]
#![feature(staged_api)]
#![feature(std_internals)]
#![feature(stdsimd)]
#![feature(stmt_expr_attributes)]
#![feature(str_internals)]
#![feature(thread_local)]
#![feature(todo_macro)]
#![feature(toowned_clone_into)]
#![feature(try_reserve)]
#![feature(unboxed_closures)]
#![feature(untagged_unions)]
#![feature(unwind_attributes)]
// NB: the above list is sorted to minimize merge conflicts.

#![default_lib_allocator]

// Explicitly import the prelude. The compiler uses this same unstable attribute
// to import the prelude implicitly when building crates that depend on std.
#[prelude_import]
#[allow(unused)]
use prelude::v1::*;

// Access to Bencher, etc.
#[cfg(test)] extern crate test;

// Re-export a few macros from core
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::{assert_eq, assert_ne, debug_assert, debug_assert_eq, debug_assert_ne};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::{unreachable, unimplemented, write, writeln, r#try, todo};

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

// Only needed for now for the `std_detect` module until that crate changes to
// use `cfg_if::cfg_if!`
#[macro_use]
#[cfg(not(test))]
extern crate cfg_if;

// During testing, this crate is not actually the "real" std library, but rather
// it links to the real std library, which was compiled from this same source
// code. So any lang items std defines are conditionally excluded (or else they
// would generate duplicate lang item errors), and any globals it defines are
// _not_ the globals used by "real" std. So this import, defined only during
// testing gives test-std access to real-std lang items and globals. See #2912
#[cfg(test)] extern crate std as realstd;

// The standard macros that are not built-in to the compiler.
#[macro_use]
mod macros;

// The Rust prelude
pub mod prelude;

// Public module declarations and re-exports
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::any;
#[stable(feature = "simd_arch", since = "1.27.0")]
#[doc(no_inline)]
pub use core::arch;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::cell;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::clone;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::cmp;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::convert;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::default;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::hash;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::intrinsics;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::iter;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::marker;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::mem;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::ops;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::ptr;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::raw;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::result;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::option;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::isize;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::i8;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::i16;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::i32;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::i64;
#[stable(feature = "i128", since = "1.26.0")]
pub use core::i128;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::usize;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::u8;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::u16;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::u32;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::u64;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::boxed;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::rc;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::borrow;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::fmt;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::format;
#[stable(feature = "pin", since = "1.33.0")]
pub use core::pin;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::slice;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::str;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::string;
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::vec;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::char;
#[stable(feature = "i128", since = "1.26.0")]
pub use core::u128;
#[stable(feature = "core_hint", since = "1.27.0")]
pub use core::hint;
#[stable(feature = "core_array", since = "1.36.0")]
pub use core::array;

pub mod f32;
pub mod f64;

#[macro_use]
pub mod thread;
pub mod ascii;
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

#[stable(feature = "futures_api", since = "1.36.0")]
pub mod task {
    //! Types and Traits for working with asynchronous tasks.
    #[doc(inline)]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub use core::task::*;
}

#[stable(feature = "futures_api", since = "1.36.0")]
pub mod future;

// Platform-abstraction modules
#[macro_use]
mod sys_common;
mod sys;

pub mod alloc;

// Private support modules
mod panicking;
mod memchr;

// The runtime entry point and a few unstable public functions used by the
// compiler
pub mod rt;

// Pull in the `std_detect` crate directly into libstd. The contents of
// `std_detect` are in a different repository: rust-lang-nursery/stdsimd.
//
// `std_detect` depends on libstd, but the contents of this module are
// set up in such a way that directly pulling it here works such that the
// crate uses the this crate as its libstd.
#[path = "../stdsimd/crates/std_detect/src/mod.rs"]
#[allow(missing_debug_implementations, missing_docs, dead_code)]
#[unstable(feature = "stdsimd", issue = "48556")]
#[cfg(not(test))]
mod std_detect;

#[doc(hidden)]
#[unstable(feature = "stdsimd", issue = "48556")]
#[cfg(not(test))]
pub use std_detect::detect;

// Include a number of private modules that exist solely to provide
// the rustdoc documentation for primitive types. Using `include!`
// because rustdoc only looks for these modules at the crate level.
include!("primitive_docs.rs");

// Include a number of private modules that exist solely to provide
// the rustdoc documentation for the existing keywords. Using `include!`
// because rustdoc only looks for these modules at the crate level.
include!("keyword_docs.rs");
