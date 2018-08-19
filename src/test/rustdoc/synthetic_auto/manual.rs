// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// @has manual/struct.Foo.html
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//*/code' 'impl<T> Sync for \
// Foo<T> where T: Sync'
//
// @has - '//*[@id="implementations-list"]/*[@class="impl"]//*/code' \
// 'impl<T> Send for Foo<T>'
//
// @count - '//*[@id="implementations-list"]/*[@class="impl"]' 1
// @count - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]' 1
pub struct Foo<T> {
    field: T,
}

unsafe impl<T> Send for Foo<T> {}
