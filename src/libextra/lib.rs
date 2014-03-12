// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Rust extras.

The `extra` crate is a set of useful modules for a variety of
purposes, including collections, numerics, I/O, serialization,
and concurrency.

Rust extras are part of the standard Rust distribution.

*/

#[crate_id = "extra#0.10-pre"];
#[comment = "Rust extras"];
#[license = "MIT/ASL2"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];
#[doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://static.rust-lang.org/doc/master")];

#[feature(macro_rules, globs, managed_boxes, asm, default_type_params)];

#[deny(non_camel_case_types)];
#[deny(missing_doc)];

extern crate collections;
extern crate rand;
extern crate serialize;
extern crate sync;
extern crate time;

// Utility modules
pub mod c_vec;
pub mod url;
pub mod tempfile;
pub mod workcache;
pub mod stats;

#[cfg(unicode)]
mod unicode;
