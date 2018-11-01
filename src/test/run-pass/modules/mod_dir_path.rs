// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
#![allow(unused_macros)]
// ignore-pretty issue #37195

mod mod_dir_simple {
    #[path = "mod_dir_simple/test.rs"]
    pub mod syrup;
}

pub fn main() {
    assert_eq!(mod_dir_simple::syrup::foo(), 10);

    mod foo {
        #[path = "auxiliary/two_macros_2.rs"]
        mod two_macros_2;
    }

    mod bar {
        macro_rules! m { () => {
            #[path = "auxiliary/two_macros_2.rs"]
            mod two_macros_2;
        } }
        m!();
    }
}
