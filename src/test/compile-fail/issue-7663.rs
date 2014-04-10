// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(globs)]
#![deny(unused_imports)]
#![allow(dead_code)]

mod test1 {

    mod foo { pub fn p() -> int { 1 } }
    mod bar { pub fn p() -> int { 2 } }

    pub mod baz {
        use test1::foo::*; //~ ERROR: unused import
        use test1::bar::*;

        pub fn my_main() { assert!(p() == 2); }
    }
}

mod test2 {

    mod foo { pub fn p() -> int { 1 } }
    mod bar { pub fn p() -> int { 2 } }

    pub mod baz {
        use test2::foo::p; //~ ERROR: unused import
        use test2::bar::p;

        pub fn my_main() { assert!(p() == 2); }
    }
}

mod test3 {

    mod foo { pub fn p() -> int { 1 } }
    mod bar { pub fn p() -> int { 2 } }

    pub mod baz {
        use test3::foo::*; //~ ERROR: unused import
        use test3::bar::p;

        pub fn my_main() { assert!(p() == 2); }
    }
}

fn main() {
}

