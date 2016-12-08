// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that debuginfo does not introduce a dependency edge to the Krate
// dep-node.

// revisions:rpass1 rpass2

#![feature(rustc_attrs)]


#![rustc_partition_translated(module="issue_38222-mod1", cfg="rpass2")]

// If trans had added a dependency edge to the Krate dep-node, nothing would
// be re-used, so checking that this module was re-used is sufficient.
#![rustc_partition_reused(module="issue_38222", cfg="rpass2")]

//[rpass1] compile-flags: -C debuginfo=1
//[rpass2] compile-flags: -C debuginfo=1

pub fn main() {
    mod1::some_fn();
}

mod mod1 {
    pub fn some_fn() {
        let _ = 1;
    }

    #[cfg(rpass2)]
    fn _some_other_fn() {
    }
}
