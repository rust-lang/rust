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

pub struct Alpha<A> where A: MyTrait;
pub trait Bravo<B> where B: MyTrait {}
pub fn charlie<C>() where C: MyTrait {}

pub struct Delta<D>;
impl<D> Delta<D> where D: MyTrait {
    pub fn delta() {}
}

pub struct Echo<E>;
impl<E> MyTrait for Echo<E> where E: MyTrait {}

pub enum Foxtrot<F> {}
impl<F> MyTrait for Foxtrot<F> where F: MyTrait {}
