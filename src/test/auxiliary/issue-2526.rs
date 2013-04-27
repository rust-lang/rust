// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name = "issue_2526",
       vers = "0.2",
       uuid = "54cc1bc9-02b8-447c-a227-75ebc923bc29")];
#[crate_type = "lib"];

extern mod std;

struct arc_destruct<T> {
  _data: int,
}

#[unsafe_destructor]
impl<T:Const> Drop for arc_destruct<T> {
    fn finalize(&self) {}
}

fn arc_destruct<T:Const>(data: int) -> arc_destruct<T> {
    arc_destruct {
        _data: data
    }
}

fn arc<T:Const>(_data: T) -> arc_destruct<T> {
    arc_destruct(0)
}

fn init() -> arc_destruct<context_res> {
    unsafe {
        arc(context_res())
    }
}

struct context_res {
    ctx : int,
}

impl Drop for context_res {
    fn finalize(&self) {}
}

fn context_res() -> context_res {
    context_res {
        ctx: 0
    }
}

pub type context = arc_destruct<context_res>;

