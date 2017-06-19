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
    pub enum Foo {
        Bar,
    }
    pub use self::Foo::*;
}

// @has 'issue_35488/index.html' '//code' 'pub use self::Foo::*;'
// @has 'issue_35488/enum.Foo.html'
pub use self::foo::*;

// @has 'issue_35488/index.html' '//code' 'pub use std::option::Option::None;'
pub use std::option::Option::None;
