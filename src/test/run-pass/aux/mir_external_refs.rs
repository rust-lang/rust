// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


pub struct S(pub u8);

impl S {
    pub fn hey() -> u8 { 24 }
}

pub trait X {
    fn hoy(&self) -> u8 { 25 }
}

impl X for S {}

pub enum E {
    U(u8)
}

pub fn regular_fn() -> u8 { 12 }
