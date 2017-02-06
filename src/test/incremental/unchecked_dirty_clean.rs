// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: rpass1 cfail2
// compile-flags: -Z query-dep-graph

#![allow(warnings)]
#![feature(rustc_attrs)]

// Sanity check for the dirty-clean system. We add #[rustc_dirty]/#[rustc_clean]
// attributes in places that are not checked and make sure that this causes an
// error.

fn main() {

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    //[cfail2]~^ ERROR found unchecked #[rustc_dirty]/#[rustc_clean] attribute
    {
        // empty block
    }

    #[rustc_clean(label="Hir", cfg="cfail2")]
    //[cfail2]~^ ERROR found unchecked #[rustc_dirty]/#[rustc_clean] attribute
    {
        // empty block
    }
}

struct _Struct {
    #[rustc_dirty(label="Hir", cfg="cfail2")]
    //[cfail2]~^ ERROR found unchecked #[rustc_dirty]/#[rustc_clean] attribute
    _field1: i32,

    #[rustc_clean(label="Hir", cfg="cfail2")]
    //[cfail2]~^ ERROR found unchecked #[rustc_dirty]/#[rustc_clean] attribute
    _field2: i32,
}
