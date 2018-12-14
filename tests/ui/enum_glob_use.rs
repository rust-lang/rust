// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::all, clippy::pedantic)]
#![allow(unused_imports, dead_code, clippy::missing_docs_in_private_items)]

use std::cmp::Ordering::*;

enum Enum {
    _Foo,
}

use self::Enum::*;

fn blarg() {
    use self::Enum::*; // ok, just for a function
}

mod blurg {
    pub use std::cmp::Ordering::*; // ok, re-export
}

mod tests {
    use super::*;
}

#[allow(non_snake_case)]
mod CamelCaseName {}

use CamelCaseName::*;

fn main() {}
