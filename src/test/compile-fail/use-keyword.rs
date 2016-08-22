// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that imports with nakes super and self don't fail during parsing
// FIXME: this shouldn't fail during name resolution either

mod a {
    mod b {
        use self as A; //~ ERROR `self` imports are only allowed within a { } list
        //~^ ERROR unresolved import `self` [E0432]
        //~| no `self` in the root
        use super as B;
        //~^ ERROR unresolved import `super` [E0432]
        //~| no `super` in the root
        use super::{self as C};
        //~^ ERROR unresolved import `super` [E0432]
        //~| no `super` in the root
    }
}

fn main() {}
