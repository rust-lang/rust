// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #35593. Check that we can reuse this trivially
// equal example.

// revisions:rpass1 rpass2

#![feature(rustc_attrs)]
#![rustc_partition_reused(module="issue_35593", cfg="rpass2")]

fn main() {
    println!("hello world");
}
