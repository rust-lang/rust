// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The Rust core library

#![crate_id = "core#0.11-pre"]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://static.rust-lang.org/doc/master")]

#![no_std]
#![feature(globs, macro_rules, managed_boxes, phase)]
#![deny(missing_doc)]

#[cfg(test)] extern crate realcore = "core";
#[cfg(test)] extern crate libc;
#[cfg(test)] extern crate native;
#[phase(syntax, link)] #[cfg(test)] extern crate realstd = "std";
#[phase(syntax, link)] #[cfg(test)] extern crate log;

#[cfg(test)] pub use cmp = realcore::cmp;
#[cfg(test)] pub use kinds = realcore::kinds;
#[cfg(test)] pub use ops = realcore::ops;
#[cfg(test)] pub use owned = realcore::owned;
#[cfg(test)] pub use ty = realcore::ty;

#[cfg(not(test))]
mod macros;

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

pub mod num;

/* The libcore prelude, not as all-encompassing as the libstd prelude */

pub mod prelude;

/* Core modules for ownership management */

pub mod cast;
pub mod intrinsics;
pub mod mem;
pub mod ptr;

/* Core language traits */

#[cfg(not(test))] pub mod kinds;
#[cfg(not(test))] pub mod ops;
#[cfg(not(test))] pub mod ty;
#[cfg(not(test))] pub mod cmp;
#[cfg(not(test))] pub mod owned;
pub mod clone;
pub mod default;
pub mod container;

/* Core types and methods on primitives */

mod unicode;
mod unit;
pub mod any;
pub mod bool;
pub mod cell;
pub mod char;
pub mod finally;
pub mod iter;
pub mod option;
pub mod raw;
pub mod result;
pub mod slice;
pub mod str;
pub mod tuple;

mod failure;

// FIXME: this module should not exist. Once owned allocations are no longer a
//        language type, this module can move outside to the owned allocation
//        crate.
mod should_not_exist;

mod std {
    pub use clone;
    pub use cmp;

    #[cfg(test)] pub use realstd::fmt;    // needed for fail!()
    #[cfg(test)] pub use realstd::rt;     // needed for fail!()
    #[cfg(test)] pub use realstd::option; // needed for assert!()
    #[cfg(test)] pub use realstd::os;     // needed for tests
}
