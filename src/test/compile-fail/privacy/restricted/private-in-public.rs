// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(pub_restricted)]

mod foo {
    struct Priv;
    mod bar {
        use foo::Priv;
        pub(super) fn f(_: Priv) {}
        pub(crate) fn g(_: Priv) {} //~ ERROR E0446
    }
}

fn main() { }
