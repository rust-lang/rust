// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link (name = "issue2378a")];
#[crate_type = "lib"];

pub enum maybe<T> { just(T), nothing }

impl <T:Clone> Index<uint,T> for maybe<T> {
    fn index(&self, _idx: &uint) -> T {
        match self {
            &just(ref t) => (*t).clone(),
            &nothing => { fail!(); }
        }
    }
}
