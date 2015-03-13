// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # The Unicode Library
//!
//! Unicode-intensive functions for `char` and `str` types.
//!
//! This crate provides a collection of Unicode-related functionality,
//! including decompositions, conversions, etc., and provides traits
//! implementing these functions for the `char` and `str` types.
//!
//! The functionality included here is only that which is necessary to
//! provide for basic string-related manipulations. This crate does not
//! (yet) aim to provide a full set of Unicode tables.

// Do not remove on snapshot creation. Needed for bootstrap. (Issue #22364)
#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "unicode"]
#![unstable(feature = "unicode")]
#![feature(lang_items)]
#![feature(staged_api)]
#![staged_api]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]
#![feature(no_std)]
#![no_std]
#![feature(core)]
#![doc(test(no_crate_inject))]

extern crate core;

// regex module
pub use tables::regex;

mod normalize;
mod tables;
mod u_str;
pub mod char;

pub mod str {
    pub use u_str::{UnicodeStr, Words, Graphemes, GraphemeIndices};
    pub use u_str::{utf8_char_width, is_utf16, Utf16Items, Utf16Item};
    pub use u_str::{utf16_items, Utf16Encoder};
}
