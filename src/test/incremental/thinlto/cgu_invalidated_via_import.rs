// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// This test checks that the LTO phase is re-done for CGUs that import something
// via ThinLTO and that imported thing changes while the definition of the CGU
// stays untouched.

// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph -O
// compile-pass

#![feature(rustc_attrs)]
#![crate_type="rlib"]

#![rustc_expected_cgu_reuse(module="cgu_invalidated_via_import-foo",
                            cfg="cfail2",
                            kind="no")]
#![rustc_expected_cgu_reuse(module="cgu_invalidated_via_import-foo",
                            cfg="cfail3",
                            kind="post-lto")]

#![rustc_expected_cgu_reuse(module="cgu_invalidated_via_import-bar",
                            cfg="cfail2",
                            kind="pre-lto")]
#![rustc_expected_cgu_reuse(module="cgu_invalidated_via_import-bar",
                            cfg="cfail3",
                            kind="post-lto")]

mod foo {

    // Trivial functions like this one are imported very reliably by ThinLTO.
    #[cfg(cfail1)]
    pub fn inlined_fn() -> u32 {
        1234
    }

    #[cfg(not(cfail1))]
    pub fn inlined_fn() -> u32 {
        1234
    }
}

pub mod bar {
    use foo::inlined_fn;

    pub fn caller() -> u32 {
        inlined_fn()
    }
}
