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
//! It is linked to all Rust crates by default.
//!
//! ## Intrinsic types and operations
//!
//! The [`ptr`](ptr/index.html) and [`mem`](mem/index.html)
//! modules deal with unsafe pointers and memory manipulation.
//! [`kinds`](kinds/index.html) defines the special built-in traits,
//! and [`raw`](raw/index.html) the runtime representation of Rust types.
//! These are some of the lowest-level building blocks in Rust.
//!
//! ## Math on primitive types and math traits
//!
//! Although basic operations on primitive types are implemented
//! directly by the compiler, the standard library additionally
//! defines many common operations through traits defined in
//! mod [`num`](num/index.html).
//!
//! ## Pervasive types
//!
//! The [`option`](option/index.html) and [`result`](result/index.html)
//! modules define optional and error-handling types, `Option` and `Result`.
//! [`iter`](iter/index.html) defines Rust's iterator protocol
//! along with a wide variety of iterators.
//! [`Cell` and `RefCell`](cell/index.html) are for creating types that
//! manage their own mutability.
//!
//! ## Vectors, slices and strings
//!
//! The common container type, `Vec`, a growable vector backed by an
//! array, lives in the [`vec`](vec/index.html) module. References to
//! arrays, `&[T]`, more commonly called "slices", are built-in types
//! for which the [`slice`](slice/index.html) module defines many
//! methods.
//!
//! `&str`, a UTF-8 string, is a built-in type, and the standard library
//! defines methods for it on a variety of traits in the
//! [`str`](str/index.html) module. Rust strings are immutable;
//! use the `String` type defined in [`string`](string/index.html)
//! for a mutable string builder.
//!
//! For converting to strings use the [`format!`](fmt/index.html)
//! macro, and for converting from strings use the
//! [`FromStr`](from_str/index.html) trait.
//!
//! ## Platform abstractions
//!
//! Besides basic data types, the standard library is largely concerned
//! with abstracting over differences in common platforms, most notably
//! Windows and Unix derivatives. The [`os`](os/index.html) module
//! provides a number of basic functions for interacting with the
//! operating environment, including program arguments, environment
//! variables, and directory navigation. The [`path`](path/index.html)
//! module encapsulates the platform-specific rules for dealing
//! with file paths.
//!
//! `std` also includes modules for interoperating with the
//! C language: [`c_str`](c_str/index.html) and
//! [`c_vec`](c_vec/index.html).
//!
//! ## Concurrency, I/O, and the runtime
//!
//! The [`task`](task/index.html) module contains Rust's threading abstractions,
//! while [`comm`](comm/index.html) contains the channel types for message
//! passing. [`sync`](sync/index.html) contains further, primitive, shared
//! memory types, including [`atomics`](sync/atomics/index.html).
//!
//! Common types of I/O, including files, TCP, UDP, pipes, Unix domain sockets,
//! timers, and process spawning, are defined in the [`io`](io/index.html) module.
//!
//! Rust's I/O and concurrency depends on a small runtime interface
//! that lives, along with its support code, in mod [`rt`](rt/index.html).
//! While a notable part of the standard library's architecture, this
//! module is not intended for public use.
//!
//! ## The Rust prelude and macros
//!
//! Finally, the [`prelude`](prelude/index.html) defines a
//! common set of traits, types, and functions that are made available
//! to all code by default. [`macros`](macros/index.html) contains
//! all the standard macros, such as `assert!`, `fail!`, `println!`,
//! and `format!`, also available to all Rust code.

#![crate_name = "std"]
#![unstable]
#![comment = "The Rust standard library"]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]

#![allow(unknown_features)]
#![feature(macro_rules, globs, linkage)]
#![feature(default_type_params, phase, lang_items, unsafe_destructor)]
#![feature(import_shadowing, slicing_syntax)]

// Don't link to std. We are std.
#![no_std]

#![deny(missing_doc)]

#![reexport_test_harness_main = "test_main"]

#[cfg(test)] extern crate green;
#[cfg(test)] extern crate debug;
#[cfg(test)] #[phase(plugin, link)] extern crate log;

extern crate alloc;
extern crate unicode;
extern crate core;
extern crate "collections" as core_collections;
extern crate "rand" as core_rand;
extern crate "sync" as core_sync;
extern crate libc;
extern crate rustrt;

// Make std testable by not duplicating lang items. See #2912
#[cfg(test)] extern crate "std" as realstd;
#[cfg(test)] pub use realstd::kinds;
#[cfg(test)] pub use realstd::ops;
#[cfg(test)] pub use realstd::cmp;
#[cfg(test)] pub use realstd::ty;
#[cfg(test)] pub use realstd::boxed;


// NB: These reexports are in the order they should be listed in rustdoc

pub use core::any;
pub use core::bool;
pub use core::cell;
pub use core::clone;
#[cfg(not(test))] pub use core::cmp;
pub use core::default;
pub use core::finally;
pub use core::intrinsics;
pub use core::iter;
#[cfg(not(test))] pub use core::kinds;
pub use core::mem;
#[cfg(not(test))] pub use core::ops;
pub use core::ptr;
pub use core::raw;
pub use core::simd;
pub use core::tuple;
// FIXME #15320: primitive documentation needs top-level modules, this
// should be `std::tuple::unit`.
pub use core::unit;
#[cfg(not(test))] pub use core::ty;
pub use core::result;
pub use core::option;

pub use alloc::boxed;
#[deprecated = "use boxed instead"]
pub use boxed as owned;

pub use alloc::rc;

pub use core_collections::slice;
pub use core_collections::str;
pub use core_collections::string;
pub use core_collections::vec;

pub use rustrt::c_str;
pub use rustrt::local_data;

pub use unicode::char;

pub use core_sync::comm;

/* Exported macros */

pub mod macros;
pub mod bitflags;

mod rtdeps;

/* The Prelude. */

pub mod prelude;


/* Primitive types */

#[path = "num/float_macros.rs"] mod float_macros;
#[path = "num/int_macros.rs"]   mod int_macros;
#[path = "num/uint_macros.rs"]  mod uint_macros;

#[path = "num/int.rs"]  pub mod int;
#[path = "num/i8.rs"]   pub mod i8;
#[path = "num/i16.rs"]  pub mod i16;
#[path = "num/i32.rs"]  pub mod i32;
#[path = "num/i64.rs"]  pub mod i64;

#[path = "num/uint.rs"] pub mod uint;
#[path = "num/u8.rs"]   pub mod u8;
#[path = "num/u16.rs"]  pub mod u16;
#[path = "num/u32.rs"]  pub mod u32;
#[path = "num/u64.rs"]  pub mod u64;

#[path = "num/f32.rs"]   pub mod f32;
#[path = "num/f64.rs"]   pub mod f64;

pub mod rand;

pub mod ascii;

pub mod time;

/* Common traits */

pub mod from_str;
pub mod num;
pub mod to_string;

/* Common data structures */

pub mod collections;
pub mod hash;

/* Tasks and communication */

pub mod task;
pub mod sync;

/* Runtime and platform support */

pub mod c_vec;
pub mod dynamic_lib;
pub mod os;
pub mod io;
pub mod path;
pub mod fmt;

// FIXME #7809: This shouldn't be pub, and it should be reexported under 'unstable'
// but name resolution doesn't work without it being pub.
pub mod rt;
mod failure;

// A curious inner-module that's not exported that contains the binding
// 'std' so that macro-expanded references to std::error and such
// can be resolved within libstd.
#[doc(hidden)]
mod std {
    // mods used for deriving
    pub use clone;
    pub use cmp;
    pub use hash;

    pub use comm; // used for select!()
    pub use fmt; // used for any formatting strings
    pub use io; // used for println!()
    pub use local_data; // used for local_data_key!()
    pub use option; // used for bitflags!{}
    pub use rt; // used for fail!()
    pub use vec; // used for vec![]

    // The test runner calls ::std::os::args() but really wants realstd
    #[cfg(test)] pub use realstd::os as os;
    // The test runner requires std::slice::Vector, so re-export std::slice just for it.
    //
    // It is also used in vec![]
    pub use slice;

    pub use boxed; // used for vec![]
}
