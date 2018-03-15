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

// revisions: rpass1 rpass2

#![feature(rustc_attrs)]

#[cfg(rpass1)]
fn foo() {
    fn bar() { }
    fn baz() { }
}

#[cfg(rpass2)]
#[rustc_clean(label="Hir", cfg="rpass2")]
#[rustc_clean(label="HirBody", cfg="rpass2")]
fn foo() {
    #[rustc_clean(label="Hir", cfg="rpass2")]
    #[rustc_clean(label="HirBody", cfg="rpass2")]
    fn baz() { } // order is different...

    #[rustc_clean(label="Hir", cfg="rpass2")]
    #[rustc_clean(label="HirBody", cfg="rpass2")]
    fn bar() { } // but that doesn't matter.

    fn bap() { } // neither does adding a new item
}

fn main() { }
