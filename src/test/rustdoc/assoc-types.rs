// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="lib"]

// @has assoc_types/Index.t.html
pub trait Index<I: ?Sized> {
    // @has - '//*[@id="Output.t"]//code' 'type Output: ?Sized'
    type Output: ?Sized;
    // @has - '//*[@id="index.v"]//code' \
    //      "fn index<'a>(&'a self, index: I) -> &'a Self::Output"
    fn index<'a>(&'a self, index: I) -> &'a Self::Output;
}

// @has assoc_types/use_output.v.html
// @has - '//*[@class="rust fn"]' '-> &T::Output'
pub fn use_output<T: Index<usize>>(obj: &T, index: usize) -> &T::Output {
    obj.index(index)
}

pub trait Feed {
    type Input;
}

// @has assoc_types/use_input.v.html
// @has - '//*[@class="rust fn"]' 'T::Input'
pub fn use_input<T: Feed>(_feed: &T, _element: T::Input) { }

// @has assoc_types/cmp_input.v.html
// @has - '//*[@class="rust fn"]' 'where T::Input: PartialEq<U::Input>'
pub fn cmp_input<T: Feed, U: Feed>(a: &T::Input, b: &U::Input) -> bool
    where T::Input: PartialEq<U::Input>
{
    a == b
}
