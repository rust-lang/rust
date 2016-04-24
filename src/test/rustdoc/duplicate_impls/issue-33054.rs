// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// @has issue_33054/impls/struct.Foo.html
// @has - '//code' 'impl Foo'
// @has - '//code' 'impl Bar for Foo'
// @count - '//*[@class="impl"]' 2
// @has issue_33054/impls/bar/trait.Bar.html
// @has - '//code' 'impl Bar for Foo'
// @count - '//*[@class="struct"]' 1
pub mod impls;

#[doc(inline)]
pub use impls as impls2;
