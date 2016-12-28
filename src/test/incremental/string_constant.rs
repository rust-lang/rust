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

// Here the only thing which changes is the string constant in `x`.
// Therefore, the compiler deduces (correctly) that typeck is not
// needed even for callers of `x`.

fn main() { }

mod x {
    #[cfg(rpass1)]
    pub fn x() {
        println!("{}", "1");
    }

    #[cfg(rpass2)]
    #[rustc_dirty(label="TypeckItemBody", cfg="rpass2")]
    #[rustc_dirty(label="TransCrateItem", cfg="rpass2")]
    pub fn x() {
        println!("{}", "2");
    }
}

mod y {
    use x;

    #[rustc_clean(label="TypeckItemBody", cfg="rpass2")]
    #[rustc_clean(label="TransCrateItem", cfg="rpass2")]
    pub fn y() {
        x::x();
    }
}

mod z {
    use y;

    #[rustc_clean(label="TypeckItemBody", cfg="rpass2")]
    #[rustc_clean(label="TransCrateItem", cfg="rpass2")]
    pub fn z() {
        y::y();
    }
}
