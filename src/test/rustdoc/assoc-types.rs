// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![crate_type="lib"]

// @has assoc_types/trait.Index.html
pub trait Index<I: ?Sized> {
    // @has - '//*[@id="associatedtype.Output"]//code' 'type Output: ?Sized'
    // @has - '//code[@id="Output.t"]' 'type Output: ?Sized'
    type Output: ?Sized;
    // @has - '//code[@id="index.v"]' 'fn index'
    // @has - '//*[@id="tymethod.index"]//code' \
    //      "fn index<'a>(&'a self, index: I) -> &'a Self::Output"
    // @has - '//*[@id="tymethod.index"]//code//a[@href="../assoc_types/trait.Index.html#associatedtype.Output"]' \
    //      "Output"
    fn index<'a>(&'a self, index: I) -> &'a Self::Output;
}

// @has assoc_types/fn.use_output.html
// @has - '//*[@class="rust fn"]' '-> &T::Output'
// @has - '//*[@class="rust fn"]//a[@href="../assoc_types/trait.Index.html#associatedtype.Output"]' 'Output'
pub fn use_output<T: Index<usize>>(obj: &T, index: usize) -> &T::Output {
    obj.index(index)
}

pub trait Feed {
    type Input;
}

// @has assoc_types/fn.use_input.html
// @has - '//*[@class="rust fn"]' 'T::Input'
// @has - '//*[@class="rust fn"]//a[@href="../assoc_types/trait.Feed.html#associatedtype.Input"]' 'Input'
pub fn use_input<T: Feed>(_feed: &T, _element: T::Input) { }

// @has assoc_types/fn.cmp_input.html
// @has - '//*[@class="rust fn"]' 'where T::Input: PartialEq<U::Input>'
// @has - '//*[@class="rust fn"]//a[@href="../assoc_types/trait.Feed.html#associatedtype.Input"]' 'Input'
pub fn cmp_input<T: Feed, U: Feed>(a: &T::Input, b: &U::Input) -> bool
    where T::Input: PartialEq<U::Input>
{
    a == b
}
