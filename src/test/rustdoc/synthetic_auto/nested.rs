// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
pub struct Inner<T> {
    field: T,
}

unsafe impl<T> Send for Inner<T>
where
    T: Copy,
{
}

// @has nested/struct.Foo.html
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//*/code' 'impl<T> Send for \
// Foo<T> where T: Copy'
//
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//*/code' \
// 'impl<T> Sync for Foo<T> where T: Sync'
pub struct Foo<T> {
    inner_field: Inner<T>,
}
