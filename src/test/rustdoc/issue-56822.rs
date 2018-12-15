// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Wrapper<T>(T);

trait MyTrait {
    type Output;
}

impl<'a, I, T: 'a> MyTrait for Wrapper<I>
    where I: MyTrait<Output=&'a T>
{
    type Output = T;
}

struct Inner<'a, T>(&'a T);

impl<'a, T> MyTrait for Inner<'a, T> {
    type Output = &'a T;
}

// @has issue_56822/struct.Parser.html
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//*/code' "impl<'a> Send for \
// Parser<'a>"
pub struct Parser<'a> {
    field: <Wrapper<Inner<'a, u8>> as MyTrait>::Output
}
