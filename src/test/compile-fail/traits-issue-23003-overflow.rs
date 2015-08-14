// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A variant of traits-issue-23003 in which an infinite series of
// types are required. This test now just compiles fine, since the
// relevant rules that triggered the overflow were removed.

#![feature(rustc_attrs)]
#![allow(dead_code)]

use std::marker::PhantomData;

trait Async {
    type Cancel;
}

struct Receipt<A:Async> {
    marker: PhantomData<A>,
}

struct Complete<B> {
    core: Option<B>,
}

impl<B> Async for Complete<B> {
    type Cancel = Receipt<Complete<Option<B>>>;
}

fn foo(_: Receipt<Complete<()>>) { }

#[rustc_error]
fn main() { } //~ ERROR compilation successful
