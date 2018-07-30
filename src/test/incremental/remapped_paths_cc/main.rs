// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions:rpass1 rpass2 rpass3
// compile-flags: -Z query-dep-graph -g
// aux-build:extern_crate.rs

// This test case makes sure that we detect if paths emitted into debuginfo
// are changed, even when the change happens in an external crate.

#![feature(rustc_attrs)]

#![rustc_partition_reused(module="main", cfg="rpass2")]
#![rustc_partition_reused(module="main-some_mod", cfg="rpass2")]
#![rustc_partition_reused(module="main", cfg="rpass3")]
#![rustc_partition_codegened(module="main-some_mod", cfg="rpass3")]

extern crate extern_crate;

fn main() {
    some_mod::some_fn();
}

mod some_mod {
    use extern_crate;

    pub fn some_fn() {
        extern_crate::inline_fn();
    }
}
