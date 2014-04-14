// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_id = "foo#0.11-pre"]

// Querying about the crate metadata should *not* parse the entire crate, it
// only needs the crate attributes (which are guaranteed to be at the top) be
// sure that if we have an error like a missing module that we can still query
// about the crate id.
mod error;
