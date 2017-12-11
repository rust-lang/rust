// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that the hash of `foo` doesn't change just because we ordered
// the nested items (or even added new ones).

// revisions: cfail1 cfail2
// must-compile-successfully

#![crate_type = "rlib"]
#![feature(rustc_attrs)]

#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
pub fn foo() {
    #[cfg(cfail1)]
    pub fn baz() { } // order is different...

    #[rustc_clean(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="HirBody", cfg="cfail2")]
    pub fn bar() { } // but that doesn't matter.

    #[cfg(cfail2)]
    pub fn baz() { } // order is different...

    pub fn bap() { } // neither does adding a new item
}
