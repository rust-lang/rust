// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: rpass1 rpass2
// compile-flags: -Z query-dep-graph

#![allow(warnings)]
#![feature(rustc_attrs)]
#![rustc_partition_reused(module="krate_inlined-x", cfg="rpass2")]

fn main() { }

mod x {
    fn method() {
        // use some methods that require inlining HIR from another crate:
        let mut v = vec![];
        v.push(1);
    }
}

#[cfg(rpass1)]
fn bar() { } // remove this unrelated fn in rpass2, which should not affect `x::method`
