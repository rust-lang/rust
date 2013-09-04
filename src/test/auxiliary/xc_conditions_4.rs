// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[crate_type="lib"];

#[deriving(Eq)]
pub enum Color {
    Red, Green, Blue
}

condition! {
    pub oops: (int,float,~str) -> Color;
}

pub trait Thunk<T> {
    fn call(self) -> T;
}

pub fn callback<T,TH:Thunk<T>>(t:TH) -> T {
    t.call()
}
