// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(optin_builtin_traits)]

pub auto trait AnOibit {}

pub struct Foo<T> { field: T }

// @has impl_parts/struct.Foo.html '//*[@class="impl"]//code' \
//     "impl<T: Clone> !AnOibit for Foo<T> where T: Sync,"
// @has impl_parts/trait.AnOibit.html '//*[@class="item-list"]//code' \
//     "impl<T: Clone> !AnOibit for Foo<T> where T: Sync,"
impl<T: Clone> !AnOibit for Foo<T> where T: Sync {}
