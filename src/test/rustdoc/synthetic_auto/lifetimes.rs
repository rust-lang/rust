// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
pub struct Inner<'a, T: 'a> {
    field: &'a T,
}

unsafe impl<'a, T> Send for Inner<'a, T>
where
    'a: 'static,
    T: for<'b> Fn(&'b bool) -> &'a u8,
{}

// @has lifetimes/struct.Foo.html
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//*/code' "impl<'c, K> Send \
// for Foo<'c, K> where K: for<'b> Fn(&'b bool) -> &'c u8, 'c: 'static"
//
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//*/code' "impl<'c, K> Sync \
// for Foo<'c, K> where K: Sync"
pub struct Foo<'c, K: 'c> {
    inner_field: Inner<'c, K>,
}
