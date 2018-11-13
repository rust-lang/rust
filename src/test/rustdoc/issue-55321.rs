// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(optin_builtin_traits)]

// @has issue_55321/struct.A.html
// @has - '//*[@id="implementations-list"]/*[@class="impl"]//*/code' "impl !Send for A"
// @has - '//*[@id="implementations-list"]/*[@class="impl"]//*/code' "impl !Sync for A"
pub struct A();

impl !Send for A {}
impl !Sync for A {}

// @has issue_55321/struct.B.html
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//*/code' "impl<T> !Send for \
// B<T>"
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//*/code' "impl<T> !Sync for \
// B<T>"
pub struct B<T: ?Sized>(A, Box<T>);
