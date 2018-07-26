// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    pub struct Foo;
}

mod bar {
    pub trait Bar{}

    pub fn bar() -> Box<Bar> {
        unimplemented!()
    }
}

// This makes the publicly accessible path
// differ from the internal one.
pub use foo::Foo;
pub use bar::{Bar, bar};
