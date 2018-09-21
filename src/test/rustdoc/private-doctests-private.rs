// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --test --document-private-items
// should-fail

// issue #30094: rustdoc runs doctests on private items even though those docs don't appear
//
// in this version, we show that passing --document-private-items causes the private doctests to
// run

mod private {
    /// Does all the work.
    ///
    /// ```
    /// panic!("oh no");
    /// ```
    pub fn real_job(a: u32, b: u32) -> u32 {
        return a + b;
    }
}

pub mod public {
    use super::private;

    /// ```
    /// // this was originally meant to link to public::function but we can't do that here
    /// assert_eq!(2+2, 4);
    /// ```
    pub fn function(a: u32, b: u32) -> u32 {
        return complex_helper(a, b);
    }

    /// Helps with stuff.
    ///
    /// ```
    /// panic!("oh no");
    /// ```
    fn complex_helper(a: u32, b: u32) -> u32 {
        return private::real_job(a, b);
    }
}
