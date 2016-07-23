// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

pub trait MyTrait { fn dummy(&self) { } }

// @has foo/Alpha.t.html '//pre' "pub struct Alpha<A>(_) where A: MyTrait"
pub struct Alpha<A>(A) where A: MyTrait;
// @has foo/Bravo.t.html '//pre' "pub trait Bravo<B> where B: MyTrait"
pub trait Bravo<B> where B: MyTrait { fn get(&self, B: B); }
// @has foo/charlie.v.html '//pre' "pub fn charlie<C>() where C: MyTrait"
pub fn charlie<C>() where C: MyTrait {}

pub struct Delta<D>(D);

// @has foo/Delta.t.html '//*[@class="impl"]//code' \
//          "impl<D> Delta<D> where D: MyTrait"
impl<D> Delta<D> where D: MyTrait {
    pub fn delta() {}
}

pub struct Echo<E>(E);

// @has foo/Echo.t.html '//*[@class="impl"]//code' \
//          "impl<E> MyTrait for Echo<E> where E: MyTrait"
// @has foo/MyTrait.t.html '//*[@id="implementors-list"]//code' \
//          "impl<E> MyTrait for Echo<E> where E: MyTrait"
impl<E> MyTrait for Echo<E> where E: MyTrait {}

pub enum Foxtrot<F> { Foxtrot1(F) }

// @has foo/Foxtrot.t.html '//*[@class="impl"]//code' \
//          "impl<F> MyTrait for Foxtrot<F> where F: MyTrait"
// @has foo/MyTrait.t.html '//*[@id="implementors-list"]//code' \
//          "impl<F> MyTrait for Foxtrot<F> where F: MyTrait"
impl<F> MyTrait for Foxtrot<F> where F: MyTrait {}

// @has foo/Golf.t.html '//pre[@class="rust typedef"]' \
//          "type Golf<T> where T: Clone = (T, T)"
pub type Golf<T> where T: Clone = (T, T);
