// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// @!has issue_32343/struct.Foo.html
// @has issue_32343/index.html
// @has - '//code' 'pub use foo::Foo'
// @!has - '//code/a' 'Foo'
#[doc(no_inline)]
pub use foo::Foo;

// @!has issue_32343/struct.Bar.html
// @has issue_32343/index.html
// @has - '//code' 'pub use foo::Bar'
// @has - '//code/a' 'Bar'
#[doc(no_inline)]
pub use foo::Bar;

mod foo {
    pub struct Foo;
    pub struct Bar;
}

pub mod bar {
    // @has issue_32343/bar/struct.Bar.html
    pub use ::foo::Bar;
}
