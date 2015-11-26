// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(dead_code)]

use std::marker;

struct Private<T>(marker::PhantomData<T>);
pub struct Public<T>(marker::PhantomData<T>);

pub trait PubTrait {
    type Output;
}

type PrivAlias = Public<i8>;

trait PrivTrait2 {
    type Alias;
}
impl PrivTrait2 for Private<isize> {
    type Alias = Public<u8>;
}

impl PubTrait for PrivAlias {
    type Output = Private<isize>; //~ WARN private type in public interface
}

impl PubTrait for <Private<isize> as PrivTrait2>::Alias {
    type Output = Private<isize>; //~ WARN private type in public interface
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
