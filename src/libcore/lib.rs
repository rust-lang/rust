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
#![feature(globs, macro_rules, managed_boxes)]
#![deny(missing_doc)]

/* Core modules for ownership management */

pub mod cast;
pub mod intrinsics;
pub mod mem;
pub mod ptr;

/* Core language traits */

pub mod kinds;
pub mod ops;
pub mod ty;
pub mod clone;
pub mod default;
pub mod container;

/* Core types and methods on primitives */

mod unit;
pub mod any;
pub mod finally;
pub mod raw;
pub mod char;
pub mod tuple;
