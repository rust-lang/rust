// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::useless_attribute)]

#[allow(dead_code)]
#[cfg_attr(feature = "cargo-clippy", allow(dead_code))]
#[cfg_attr(feature = "cargo-clippy", allow(dead_code))]
#[allow(unused_imports)]
#[allow(unused_extern_crates)]
#[macro_use]
extern crate clippy_lints;

// don't lint on unused_import for `use` items
#[allow(unused_imports)]
use std::collections;

// don't lint on deprecated for `use` items
mod foo {
    #[deprecated]
    pub struct Bar;
}
#[allow(deprecated)]
pub use foo::Bar;

fn main() {}
