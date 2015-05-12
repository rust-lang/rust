// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # The Rust Standard Library
//!
//! The Rust Standard Library provides the essential runtime
//! functionality for building portable Rust software.
//!
//! The rust standard library is available to all rust crates by
//! default, just as if contained an `extern crate std` import at the
//! crate root. Therefore the standard library can be accessed in
//! `use` statements through the path `std`, as in `use std::thread`,
//! or in expressions through the absolute path `::std`, as in
//! `::std::thread::sleep_ms(100)`.
//!
//! Furthermore, the standard library defines [The Rust
//! Prelude](prelude/index.html), a small collection of items, mostly
//! traits, that are imported into and available in every module.
//!
//! ## What is in the standard library
//!
//! The standard library is minimal, a set of battle-tested
//! core types and shared abstractions for the [broader Rust
//! ecosystem][https://crates.io] to build on.
//!
//! The [primitive types](#primitives), though not defined in the
//! standard library, are documented here, as are the predefined
//! [macros](#macros).
//!
//! ## Containers and collections
//!
//! The [`option`](option/index.html) and
//! [`result`](result/index.html) modules define optional and
//! error-handling types, `Option` and `Result`. The
//! [`iter`](iter/index.html) module defines Rust's iterator trait,
//! [`Iterater`](iter/trait.Iterator.html), which works with the `for`
//! loop to access collections.
//!
//! The common container type, `Vec`, a growable vector backed by an array,
//! lives in the [`vec`](vec/index.html) module. Contiguous, unsized regions
//! of memory, `[T]`, commonly called "slices", and their borrowed versions,
//! `&[T]`, commonly called "borrowed slices", are built-in types for which the
//! [`slice`](slice/index.html) module defines many methods.
//!
//! `&str`, a UTF-8 string, is a built-in type, and the standard library
//! defines methods for it on a variety of traits in the
//! [`str`](str/index.html) module. Rust strings are immutable;
//! use the `String` type defined in [`string`](string/index.html)
//! for a mutable string builder.
//!
//! For converting to strings use the [`format!`](fmt/index.html)
//! macro, and for converting from strings use the
//! [`FromStr`](str/trait.FromStr.html) trait.
//!
//! Data may be shared by placing it in a reference-counted box or the
//! [`Rc`][rc/index.html] type, and if further contained in a [`Cell`
//! or `RefCell`](cell/index.html), may be mutated as well as shared.
//! Likewise, in a concurrent setting it is common to pair an
//! atomically-reference-counted box, [`Arc`](sync/struct.Arc.html),
//! with a [`Mutex`](sync/struct.Mutex.html) to get the same effect.
//!
//! The [`collections`](collections/index.html) module defines maps,
//! sets, linked lists and other typical collection types, including
//! the common [`HashMap`](collections/struct.HashMap.html).
//!
//! ## Platform abstractions and I/O
//!
//! Besides basic data types, the standard library is largely concerned
//! with abstracting over differences in common platforms, most notably
//! Windows and Unix derivatives.
//!
//! Common types of I/O, including [files](fs/struct.File.html),
//! [TCP](net/struct.TcpStream.html),
//! [UDP](net/struct.UdpSocket.html), are defined in the
//! [`io`](io/index.html), [`fs`](fs/index.html), and
//! [`net`](net/index.html) modules.
//!
//! The [`thread`](thread/index.html) module contains Rust's threading
//! abstractions. [`sync`](sync/index.html) contains further
//! primitive shared memory types, including
//! [`atomic`](sync/atomic/index.html) and
//! [`mpsc`](sync/mpsc/index.html), which contains the channel types
//! for message passing.

// Do not remove on snapshot creation. Needed for bootstrap. (Issue #22364)
#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "std"]
#![stable(feature = "rust1", since = "1.0.0")]
#![staged_api]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]
#![doc(test(no_crate_inject, attr(deny(warnings))))]
#![doc(test(attr(allow(dead_code, deprecated, unused_variables, unused_mut))))]

#![feature(alloc)]
#![feature(allow_internal_unstable)]
#![feature(associated_consts)]
#![feature(box_syntax)]
#![feature(collections)]
#![feature(core)]
#![feature(debug_builders)]
#![feature(into_cow)]
#![feature(lang_items)]
#![feature(libc)]
#![feature(linkage, thread_local, asm)]
#![feature(macro_reexport)]
#![feature(optin_builtin_traits)]
#![feature(rand)]
#![feature(slice_patterns)]
#![feature(staged_api)]
#![feature(std_misc)]
#![feature(str_char)]
#![feature(unboxed_closures)]
#![feature(unicode)]
#![feature(unique)]
#![feature(unsafe_no_drop_flag, filling_drop)]
#![feature(zero_one)]
#![cfg_attr(test, feature(float_from_str_radix))]
#![cfg_attr(test, feature(test, rustc_private, std_misc))]

// Don't link to std. We are std.
#![feature(no_std)]
#![no_std]

#![allow(trivial_casts)]
#![deny(missing_docs)]

#[cfg(test)] extern crate test;
#[cfg(test)] #[macro_use] extern crate log;

#[macro_use]
#[macro_reexport(assert, assert_eq, debug_assert, debug_assert_eq,
    unreachable, unimplemented, write, writeln)]
extern crate core;

#[macro_use]
#[macro_reexport(vec, format)]
extern crate collections as core_collections;

#[allow(deprecated)] extern crate rand as core_rand;
extern crate alloc;
extern crate rustc_unicode;
extern crate libc;

#[macro_use] #[no_link] extern crate rustc_bitflags;

// Make std testable by not duplicating lang items. See #2912
#[cfg(test)] extern crate std as realstd;
#[cfg(test)] pub use realstd::marker;
#[cfg(test)] pub use realstd::ops;
#[cfg(test)] pub use realstd::cmp;
#[cfg(test)] pub use realstd::boxed;


// NB: These reexports are in the order they should be listed in rustdoc

pub use core::any;
pub use core::cell;
pub use core::clone;
#[cfg(not(test))] pub use core::cmp;
pub use core::convert;
pub use core::default;
pub use core::hash;
pub use core::intrinsics;
pub use core::iter;
#[cfg(not(test))] pub use core::marker;
pub use core::mem;
#[cfg(not(test))] pub use core::ops;
pub use core::ptr;
pub use core::raw;
pub use core::simd;
pub use core::result;
pub use core::option;
pub mod error;

#[cfg(not(test))] pub use alloc::boxed;
pub use alloc::rc;

pub use core_collections::borrow;
pub use core_collections::fmt;
pub use core_collections::slice;
pub use core_collections::str;
pub use core_collections::string;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core_collections::vec;

pub use rustc_unicode::char;

/* Exported macros */

#[macro_use]
mod macros;

mod rtdeps;

/* The Prelude. */

pub mod prelude;


/* Primitive types */

// NB: slice and str are primitive types too, but their module docs + primitive doc pages
// are inlined from the public re-exports of core_collections::{slice, str} above.

#[path = "num/float_macros.rs"]
#[macro_use]
mod float_macros;

#[path = "num/int_macros.rs"]
#[macro_use]
mod int_macros;

#[path = "num/uint_macros.rs"]
#[macro_use]
mod uint_macros;

#[path = "num/isize.rs"]  pub mod isize;
#[path = "num/i8.rs"]   pub mod i8;
#[path = "num/i16.rs"]  pub mod i16;
#[path = "num/i32.rs"]  pub mod i32;
#[path = "num/i64.rs"]  pub mod i64;

#[path = "num/usize.rs"] pub mod usize;
#[path = "num/u8.rs"]   pub mod u8;
#[path = "num/u16.rs"]  pub mod u16;
#[path = "num/u32.rs"]  pub mod u32;
#[path = "num/u64.rs"]  pub mod u64;

#[path = "num/f32.rs"]   pub mod f32;
#[path = "num/f64.rs"]   pub mod f64;

pub mod ascii;

pub mod thunk;

/* Common traits */

pub mod num;

/* Runtime and platform support */

#[macro_use]
pub mod thread;

pub mod collections;
pub mod dynamic_lib;
pub mod env;
pub mod ffi;
pub mod fs;
pub mod io;
pub mod net;
pub mod os;
pub mod path;
pub mod process;
pub mod sync;
pub mod time;

#[macro_use]
#[path = "sys/common/mod.rs"] mod sys_common;

#[cfg(unix)]
#[path = "sys/unix/mod.rs"] mod sys;
#[cfg(windows)]
#[path = "sys/windows/mod.rs"] mod sys;

pub mod rt;
mod panicking;
mod rand;

// Some external utilities of the standard library rely on randomness (aka
// rustc_back::TempDir and tests) and need a way to get at the OS rng we've got
// here. This module is not at all intended for stabilization as-is, however,
// but it may be stabilized long-term. As a result we're exposing a hidden,
// unstable module so we can get our build working.
#[doc(hidden)]
#[unstable(feature = "rand")]
pub mod __rand {
    pub use rand::{thread_rng, ThreadRng, Rng};
}

// Modules that exist purely to document + host impl docs for primitive types

mod array;
mod bool;
mod unit;
mod tuple;

// A curious inner-module that's not exported that contains the binding
// 'std' so that macro-expanded references to std::error and such
// can be resolved within libstd.
#[doc(hidden)]
mod std {
    pub use sync; // used for select!()
    pub use error; // used for try!()
    pub use fmt; // used for any formatting strings
    pub use option; // used for thread_local!{}
    pub use rt; // used for panic!()
    pub use vec; // used for vec![]
    pub use cell; // used for tls!
    pub use thread; // used for thread_local!
    pub use marker;  // used for tls!

    // The test runner calls ::std::env::args() but really wants realstd
    #[cfg(test)] pub use realstd::env as env;
    // The test runner requires std::slice::Vector, so re-export std::slice just for it.
    //
    // It is also used in vec![]
    pub use slice;

    pub use boxed; // used for vec![]
}
