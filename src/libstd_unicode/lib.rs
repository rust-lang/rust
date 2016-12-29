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

#![crate_name = "std_unicode"]
#![unstable(feature = "unicode", issue = "27783")]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/",
       issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/",
       test(no_crate_inject, attr(allow(unused_variables), deny(warnings))))]
#![deny(warnings)]
#![no_std]

#![feature(char_escape_debug)]
#![feature(core_char_ext)]
#![feature(decode_utf8)]
#![feature(fused)]
#![feature(lang_items)]
#![feature(staged_api)]
#![feature(try_from)]

mod tables;
mod u_str;
pub mod char;

#[allow(deprecated)]
pub mod str {
    pub use u_str::{SplitWhitespace, UnicodeStr};
    pub use u_str::{is_utf16, utf8_char_width};
    pub use u_str::Utf16Encoder;
}

// For use in libcollections, not re-exported in libstd.
pub mod derived_property {
    pub use tables::derived_property::{Case_Ignorable, Cased};
}

// For use in libsyntax
pub mod property {
    pub use tables::property::Pattern_White_Space;
}
