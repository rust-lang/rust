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
#[doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://static.rust-lang.org/doc/master")];

#[feature(macro_rules, globs, managed_boxes)];

#[deny(non_camel_case_types)];
#[deny(missing_doc)];

// Utility modules

pub mod c_vec;

// Concurrency

pub mod sync;
pub mod arc;
pub mod comm;
pub mod future;
pub mod task_pool;

// Collections

pub mod container;
pub mod bitv;
pub mod list;
pub mod ringbuf;
pub mod priority_queue;
pub mod smallintmap;

pub mod dlist;
pub mod treemap;
pub mod btree;
pub mod lru_cache;

// And ... other stuff

pub mod url;
pub mod ebml;
pub mod getopts;
pub mod json;
pub mod tempfile;
pub mod time;
pub mod base64;
pub mod workcache;
pub mod enum_set;
#[path="num/bigint.rs"]
pub mod bigint;
#[path="num/rational.rs"]
pub mod rational;
#[path="num/complex.rs"]
pub mod complex;
pub mod stats;
pub mod hex;

#[cfg(unicode)]
mod unicode;

// Compiler support modules

pub mod test;
pub mod serialize;

// A curious inner-module that's not exported that contains the binding
// 'extra' so that macro-expanded references to extra::serialize and such
// can be resolved within libextra.
#[doc(hidden)]
pub mod extra {
    pub use serialize;
    pub use test;
}
