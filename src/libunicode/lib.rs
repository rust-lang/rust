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

#![crate_id = "unicode#0.11.0"]
#![crate_name = "unicode"]
#![experimental]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/",
       html_playground_url = "http://play.rust-lang.org/")]
#![no_std]
#![allow(unused_attribute)] // NOTE: remove after stage0

extern crate core;

pub use tables::normalization::canonical_combining_class;
pub use tables::regex;

pub use u_char::UnicodeChar;
pub use u_str::UnicodeStrSlice;
pub use u_str::Words;

mod decompose;
mod tables;
mod u_char;
mod u_str;

// re-export char so that std et al see it correctly
/// Character manipulation (`char` type, Unicode Scalar Value)
///
/// This module  provides the `Char` and `UnicodeChar` traits, as well as their
/// implementation for the primitive `char` type, in order to allow basic character
/// manipulation.
///
/// A `char` actually represents a
/// *[Unicode Scalar Value](http://www.unicode.org/glossary/#unicode_scalar_value)*,
/// as it can contain any Unicode code point except high-surrogate and
/// low-surrogate code points.
///
/// As such, only values in the ranges \[0x0,0xD7FF\] and \[0xE000,0x10FFFF\]
/// (inclusive) are allowed. A `char` can always be safely cast to a `u32`;
/// however the converse is not always true due to the above range limits
/// and, as such, should be performed via the `from_u32` function..
pub mod char {
    pub use core::char::{MAX, from_u32, is_digit_radix, to_digit};
    pub use core::char::{from_digit, escape_unicode, escape_default};
    pub use core::char::{len_utf8_bytes, Char};

    pub use decompose::decompose_canonical;
    pub use decompose::decompose_compatible;

    pub use u_char::{is_alphabetic, is_XID_start, is_XID_continue};
    pub use u_char::{is_lowercase, is_uppercase, is_whitespace};
    pub use u_char::{is_alphanumeric, is_control, is_digit};
    pub use u_char::{to_uppercase, to_lowercase, width, UnicodeChar};
}
