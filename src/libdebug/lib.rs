// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Debugging utilities for Rust programs
//!
//! This crate is intended to provide useful functionality when debugging
//! programs, such as reflection for printing values. This crate is currently
//! entirely experimental as its makeup will likely change over time.
//! Additionally, it is not guaranteed that functionality such as reflection
//! will persist into the future.

#![crate_name = "debug"]
#![experimental]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/0.12.0/")]
#![experimental]
#![feature(macro_rules)]
#![allow(experimental)]

pub mod fmt;
pub mod reflect;
pub mod repr;
