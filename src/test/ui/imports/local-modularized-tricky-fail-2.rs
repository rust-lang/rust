// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// `#[macro_export] macro_rules` that doen't originate from macro expansions can be placed
// into the root module soon enough to act as usual items and shadow globs and preludes.

#![feature(decl_macro)]

// `macro_export` shadows globs
use inner1::*;

mod inner1 {
    pub macro exported() {}
}

exported!();

mod deep {
    fn deep() {
        type Deeper = [u8; {
            #[macro_export]
            macro_rules! exported {
                () => ( struct Б; ) //~ ERROR non-ascii idents are not fully supported
            }

            0
        }];
    }
}

// `macro_export` shadows std prelude
fn main() {
    panic!();
}

mod inner3 {
    #[macro_export]
    macro_rules! panic {
        () => ( struct Г; ) //~ ERROR non-ascii idents are not fully supported
    }
}

// `macro_export` shadows builtin macros
include!();

mod inner4 {
    #[macro_export]
    macro_rules! include {
        () => ( struct Д; ) //~ ERROR non-ascii idents are not fully supported
    }
}
