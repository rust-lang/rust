// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:intra-link-extern-crate.rs

// When loading `extern crate` statements, we would pull in their docs at the same time, even
// though they would never actually get displayed. This tripped intra-doc-link resolution failures,
// for items that aren't under our control, and not actually getting documented!

#![deny(intra_doc_link_resolution_failure)]

extern crate inner;
