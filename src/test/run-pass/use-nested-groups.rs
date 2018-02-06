// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(use_nested_groups)]

mod a {
    pub enum B {}

    pub mod d {
        pub enum E {}
        pub enum F {}

        pub mod g {
            pub enum H {}
            pub enum I {}
        }
    }
}

use a::{B, d::{self, *, g::H}};

fn main() {
    let _: B;
    let _: E;
    let _: F;
    let _: H;
    let _: d::g::I;
}
