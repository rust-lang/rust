// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:baz.rs
// compile-flags:--extern baz
// edition:2018

#![feature(uniform_paths)]

mod foo {
    pub type Bar = u32;
}

mod bazz {
    use foo::Bar; //~ ERROR unresolved import `foo`

    fn baz() {
        let x: Bar = 22;
    }
}

use foo::Bar;

use foobar::Baz; //~ ERROR unresolved import `foobar`

fn main() { }
