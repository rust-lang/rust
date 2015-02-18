// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name="issue_2526"]
#![crate_type = "lib"]

#![feature(unsafe_destructor)]

use std::marker;

struct arc_destruct<T> {
    _data: int,
    _marker: marker::PhantomData<T>
}

#[unsafe_destructor]
impl<T: Sync> Drop for arc_destruct<T> {
    fn drop(&mut self) {}
}

fn arc_destruct<T: Sync>(data: int) -> arc_destruct<T> {
    arc_destruct {
        _data: data,
        _marker: marker::PhantomData
    }
}

fn arc<T: Sync>(_data: T) -> arc_destruct<T> {
    arc_destruct(0)
}

fn init() -> arc_destruct<context_res> {
    arc(context_res())
}

struct context_res {
    ctx : int,
}

impl Drop for context_res {
    fn drop(&mut self) {}
}

fn context_res() -> context_res {
    context_res {
        ctx: 0
    }
}

pub type context = arc_destruct<context_res>;
