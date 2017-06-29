// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod a {}

macro_rules! m {
    () => {
        use a::$crate; //~ ERROR unresolved import `a::$crate`
        use a::$crate::b; //~ ERROR unresolved import `a::$crate::b`
        type A = a::$crate; //~ ERROR cannot find type `$crate` in module `a`
    }
}

m!();

fn main() {}
