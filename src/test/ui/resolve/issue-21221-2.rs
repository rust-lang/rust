// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod foo {
    pub mod bar {
        // note: trait T is not public, but being in the current
        // crate, it's fine to show it, since the programmer can
        // decide to make it public based on the suggestion ...
        pub trait T {}
    }
    // imports should be ignored:
    use self::bar::T;
}

pub mod baz {
    pub use foo;
    pub use std::ops::{Mul as T};
}

struct Foo;
impl T for Foo { }
//~^ ERROR unresolved trait `T`
//~| HELP you can import it into scope: `use foo::bar::T;`
