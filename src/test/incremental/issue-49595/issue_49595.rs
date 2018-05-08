// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions:cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph --test
// compile-pass

#![feature(rustc_attrs)]
#![crate_type = "rlib"]

#![rustc_partition_codegened(module="issue_49595-tests", cfg="cfail2")]
#![rustc_partition_codegened(module="issue_49595-lit_test", cfg="cfail3")]

mod tests {
    #[cfg_attr(not(cfail1), ignore)]
    #[test]
    fn test() {
    }
}


// Checks that changing a string literal without changing its span
// takes effect.

// replacing a module to have a stable span
#[cfg_attr(not(cfail3), path = "auxiliary/lit_a.rs")]
#[cfg_attr(cfail3, path = "auxiliary/lit_b.rs")]
mod lit;

pub mod lit_test {
    #[test]
    fn lit_test() {
        println!("{}", ::lit::A);
    }
}
