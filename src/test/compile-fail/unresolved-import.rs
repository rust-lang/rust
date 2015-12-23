// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

use foo::bar; //~ ERROR unresolved import `foo::bar`. Maybe a missing `extern crate foo`?

use bar::Baz as x; //~ ERROR unresolved import `bar::Baz`. There is no `Baz` in `bar`. Did you mean to use `Bar`?

use food::baz; //~ ERROR unresolved import `food::baz`. There is no `baz` in `food`. Did you mean to use the re-exported import `bag`?

use food::{beens as Foo}; //~ ERROR unresolved import `food::beens`. There is no `beens` in `food`. Did you mean to use the re-exported import `beans`?

mod bar {
    pub struct Bar;
}

mod food {
    pub use self::zug::baz::{self as bag, foobar as beans};

    mod zug {
        pub mod baz {
            pub struct foobar;
        }
    }
}
