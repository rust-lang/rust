// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// test that invoking functions which require
// dictionaries from inside an @fn works
// (at one point, it didn't)

fn mk_nil<C:ty_ops>(cx: C) -> uint {
    cx.mk()
}

trait ty_ops {
    fn mk(&self) -> uint;
}

impl ty_ops for () {
    fn mk(&self) -> uint { 22u }
}

pub fn main() {
    let fn_env: @fn() -> uint = || mk_nil(());
    assert_eq!(fn_env(), 22u);
}
