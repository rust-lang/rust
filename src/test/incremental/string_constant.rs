// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: cfail1 cfail2
// compile-flags: -Z query-dep-graph
// must-compile-successfully

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type = "rlib"]

// Here the only thing which changes is the string constant in `x`.
// Therefore, the compiler deduces (correctly) that typeck is not
// needed even for callers of `x`.


pub mod x {
    #[cfg(cfail1)]
    pub fn x() {
        println!("{}", "1");
    }

    #[cfg(cfail2)]
    #[rustc_dirty(label="HirBody", cfg="cfail2")]
    #[rustc_dirty(label="MirOptimized", cfg="cfail2")]
    pub fn x() {
        println!("{}", "2");
    }
}

pub mod y {
    use x;

    #[rustc_clean(label="TypeckTables", cfg="cfail2")]
    #[rustc_clean(label="MirOptimized", cfg="cfail2")]
    pub fn y() {
        x::x();
    }
}

pub mod z {
    use y;

    #[rustc_clean(label="TypeckTables", cfg="cfail2")]
    #[rustc_clean(label="MirOptimized", cfg="cfail2")]
    pub fn z() {
        y::y();
    }
}
