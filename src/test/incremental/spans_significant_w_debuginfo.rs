// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test makes sure that just changing a definition's location in the
// source file also changes its incr. comp. hash, if debuginfo is enabled.

// revisions:rpass1 rpass2

// compile-flags: -g -Z query-dep-graph

#![feature(rustc_attrs)]

#[cfg(rpass1)]
pub fn main() {}

#[cfg(rpass2)]
#[rustc_dirty(label="Hir", cfg="rpass2")]
#[rustc_dirty(label="HirBody", cfg="rpass2")]
pub fn main() {}
