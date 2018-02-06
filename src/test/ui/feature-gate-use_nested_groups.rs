// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_imports, dead_code)]

mod a {
    pub enum B {}
    pub enum C {}

    pub mod d {
        pub enum E {}
        pub enum F {}

        pub mod g {
            pub enum H {}
        }
    }
}

use a::{B, d::{*, g::H}};  //~ ERROR glob imports in `use` groups are experimental
                           //~^ ERROR nested groups in `use` are experimental
                           //~^^ ERROR paths in `use` groups are experimental

fn main() {}
