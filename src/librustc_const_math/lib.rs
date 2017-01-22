// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Rusty Mathematics
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![crate_name = "rustc_const_math"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]


#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(const_fn)]
#![cfg_attr(not(stage0), feature(i128))]

extern crate syntax;

// SNAP: remove use of this crate
extern crate rustc_i128;

extern crate serialize as rustc_serialize; // used by deriving

mod float;
mod int;
mod us;
mod is;
mod err;

pub use float::*;
pub use int::*;
pub use us::*;
pub use is::*;
pub use err::{ConstMathErr, Op};
