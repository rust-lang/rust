// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![feature(use_extern_macros)]

#[macro_export(local_inner_macros)]
macro_rules! dollar_crate_exported {
    (1) => { $crate::exported!(); };
    (2) => { exported!(); };
}

// Before `exported` is defined
exported!();

mod inner {

    ::exported!();
    crate::exported!();
    dollar_crate_exported!(1);
    dollar_crate_exported!(2);

    mod inner_inner {
        #[macro_export]
        macro_rules! exported {
            () => ()
        }
    }

    // After `exported` is defined
    ::exported!();
    crate::exported!();
    dollar_crate_exported!(1);
    dollar_crate_exported!(2);
}

exported!();

fn main() {}
