// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(conservative_impl_trait)]

pub fn g() -> impl Iterator<Item=u8> {
    Some(1u8).into_iter()
}

pub fn h() -> (impl Iterator<Item=u8>) {
    Some(1u8).into_iter()
}

pub fn i() -> impl Iterator<Item=u8> + 'static {
    Some(1u8).into_iter()
}

pub fn j() -> impl Iterator<Item=u8> + Clone {
    Some(1u8).into_iter()
}

// @has issue_43869/fn.g.html
// @has issue_43869/fn.h.html
// @has issue_43869/fn.i.html
// @has issue_43869/fn.j.html
