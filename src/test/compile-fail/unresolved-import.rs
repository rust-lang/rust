// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use foo::bar; //~ ERROR unresolved import `foo::bar`. Maybe a missing `extern crate foo`?

use bar::baz as x; //~ ERROR unresolved import `bar::baz`. There is no `baz` in `bar`

use food::baz; //~ ERROR unresolved import `food::baz`. There is no `baz` in `food`

use food::{quux as beans}; //~ ERROR unresolved import `food::quux`. There is no `quux` in `food`

mod bar {
    struct bar;
}

mod food {
    pub use self::zug::baz::{self as bag, quux as beans};

    mod zug {
        pub mod baz {
            pub struct quux;
        }
    }
}
