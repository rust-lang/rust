// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# The Rust standard library

The Rust standard library is a group of interrelated modules defining
the core language traits, operations on built-in data types, collections,
platform abstractions, the task scheduler, runtime support for language
features and other common functionality.

`std` includes modules corresponding to each of the integer types,
each of the floating point types, the `bool` type, tuples, characters,
strings (`str`), vectors (`vec`), managed boxes (`managed`), owned
boxes (`owned`), and unsafe and borrowed pointers (`ptr`, `borrowed`).
Additionally, `std` provides pervasive types (`option` and `result`),
task creation and communication primitives (`task`, `comm`), platform
abstractions (`os` and `path`), basic I/O abstractions (`io`), common
traits (`kinds`, `ops`, `cmp`, `num`, `to_str`), and complete bindings
to the C standard library (`libc`).

# Standard library injection and the Rust prelude

`std` is imported at the topmost level of every crate by default, as
if the first line of each crate was

    extern mod std;

This means that the contents of std can be accessed from any context
with the `std::` path prefix, as in `use std::vec`, `use std::task::spawn`,
etc.

Additionally, `std` contains a `prelude` module that reexports many of the
most common types, traits and functions. The contents of the prelude are
imported into every *module* by default.  Implicitly, all modules behave as if
they contained the following prologue:

    use std::prelude::*;

*/


#[link(name = "std",
       vers = "0.9-pre",
       uuid = "c70c24a7-5551-4f73-8e37-380b11d80be8",
       url = "https://github.com/mozilla/rust/tree/master/src/libstd")];

#[comment = "The Rust standard library"];
#[license = "MIT/ASL2"];
#[crate_type = "lib"];

#[doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://static.rust-lang.org/doc/master")];

#[feature(macro_rules, globs, asm, managed_boxes)];

// Don't link to std. We are std.
#[no_std];

#[deny(non_camel_case_types)];
#[deny(missing_doc)];

// Make extra accessible for benchmarking
#[cfg(test)] extern mod extra(vers="0.9-pre");

// Make std testable by not duplicating lang items. See #2912
#[cfg(test)] extern mod realstd(name = "std");
#[cfg(test)] pub use kinds = realstd::kinds;
#[cfg(test)] pub use ops = realstd::ops;
#[cfg(test)] pub use cmp = realstd::cmp;

// On Linux, link to the runtime with -lrt.
#[cfg(target_os = "linux")]
#[doc(hidden)]
pub mod linkhack {
    #[link_args="-lrustrt -lrt"]
    #[link_args = "-lpthread"]
    extern {
    }
}

/* The Prelude. */

pub mod prelude;

/* Primitive types */

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

pub mod unit;
pub mod bool;
pub mod char;
pub mod tuple;

pub mod vec;
pub mod at_vec;
pub mod str;

#[path = "str/ascii.rs"]
pub mod ascii;
pub mod send_str;

pub mod ptr;
pub mod owned;
pub mod managed;
pub mod borrow;
pub mod rc;


/* Core language traits */

#[cfg(not(test))] pub mod kinds;
#[cfg(not(test))] pub mod ops;
#[cfg(not(test))] pub mod cmp;


/* Common traits */

pub mod from_str;
#[path = "num/num.rs"]
pub mod num;
pub mod iter;
pub mod to_str;
pub mod to_bytes;
pub mod clone;
pub mod io;
pub mod hash;
pub mod container;
pub mod default;

/* Common data structures */

pub mod option;
pub mod result;
pub mod either;
pub mod hashmap;
pub mod cell;
pub mod trie;


/* Tasks and communication */

pub mod task;
pub mod comm;
pub mod select;
pub mod local_data;


/* Runtime and platform support */

pub mod libc;
pub mod c_str;
pub mod os;
pub mod path;
pub mod rand;
pub mod run;
pub mod sys;
pub mod cast;
pub mod fmt;
pub mod repr;
pub mod cleanup;
pub mod reflect;
pub mod condition;
pub mod logging;
pub mod util;
pub mod routine;
pub mod mem;

/* Unsupported interfaces */

// Private APIs
pub mod unstable;

/* For internal use, not exported */

mod unicode;
#[path = "num/cmath.rs"]
mod cmath;

// FIXME #7809: This shouldn't be pub, and it should be reexported under 'unstable'
// but name resolution doesn't work without it being pub.
pub mod rt;

// A curious inner-module that's not exported that contains the binding
// 'std' so that macro-expanded references to std::error and such
// can be resolved within libstd.
#[doc(hidden)]
mod std {
    pub use clone;
    pub use cmp;
    pub use condition;
    pub use option;
    pub use kinds;
    pub use local_data;
    pub use logging;
    pub use sys;
    pub use unstable;
    pub use str;
    pub use os;
    pub use fmt;
    pub use to_bytes;
    pub use logging;
}
