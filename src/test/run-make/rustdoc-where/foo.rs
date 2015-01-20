// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait MyTrait {}

// @matches foo/struct.Alpha.html '//pre' "Alpha.*where.*A:.*MyTrait"
pub struct Alpha<A> where A: MyTrait;
// @matches foo/trait.Bravo.html '//pre' "Bravo.*where.*B:.*MyTrait"
pub trait Bravo<B> where B: MyTrait {}
// @matches foo/fn.charlie.html '//pre' "charlie.*where.*C:.*MyTrait"
pub fn charlie<C>() where C: MyTrait {}

pub struct Delta<D>;
// @matches foo/struct.Delta.html '//*[@class="impl"]//code' "impl.*Delta.*where.*D:.*MyTrait"
impl<D> Delta<D> where D: MyTrait {
    pub fn delta() {}
}

pub struct Echo<E>;
// @matches foo/struct.Echo.html '//*[@class="impl"]//code' \
//          "impl.*MyTrait.*for.*Echo.*where.*E:.*MyTrait"
// @matches foo/trait.MyTrait.html '//*[@id="implementors-list"]//code' \
//          "impl.*MyTrait.*for.*Echo.*where.*E:.*MyTrait"
impl<E> MyTrait for Echo<E> where E: MyTrait {}

pub enum Foxtrot<F> {}
// @matches foo/enum.Foxtrot.html '//*[@class="impl"]//code' \
//          "impl.*MyTrait.*for.*Foxtrot.*where.*F:.*MyTrait"
// @matches foo/trait.MyTrait.html '//*[@id="implementors-list"]//code' \
//          "impl.*MyTrait.*for.*Foxtrot.*where.*F:.*MyTrait"
impl<F> MyTrait for Foxtrot<F> where F: MyTrait {}
