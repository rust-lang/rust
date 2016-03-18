// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(hidden)]
pub mod foo {
    pub struct Foo;
}

mod bar {
    pub use self::bar::Bar;
    mod bar {
        pub struct Bar;
    }
}

// @has issue_28537/struct.Foo.html
pub use foo::Foo;

// @has issue_28537/struct.Bar.html
pub use self::bar::Bar;
