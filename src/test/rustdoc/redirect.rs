// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:reexp_stripped.rs
// build-aux-docs
// ignore-cross-compile

extern crate reexp_stripped;

pub trait Foo {}

// @has redirect/index.html
// @has - '//code' 'pub use reexp_stripped::Bar'
// @has - '//code/a' 'Bar'
// @has reexp_stripped/hidden/struct.Bar.html
// @has - '//p/a' '../../reexp_stripped/struct.Bar.html'
// @has 'reexp_stripped/struct.Bar.html'
#[doc(no_inline)]
pub use reexp_stripped::Bar;
impl Foo for Bar {}

// @has redirect/index.html
// @has - '//code' 'pub use reexp_stripped::Quz'
// @has - '//code/a' 'Quz'
// @has reexp_stripped/private/struct.Quz.html
// @has - '//p/a' '../../reexp_stripped/struct.Quz.html'
// @has 'reexp_stripped/struct.Quz.html'
#[doc(no_inline)]
pub use reexp_stripped::Quz;
impl Foo for Quz {}

mod private_no_inline {
    pub struct Qux;
    impl ::Foo for Qux {}
}

// @has redirect/index.html
// @has - '//code' 'pub use private_no_inline::Qux'
// @!has - '//a' 'Qux'
// @!has redirect/struct.Qux.html
#[doc(no_inline)]
pub use private_no_inline::Qux;
