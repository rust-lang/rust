// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link (name = "issue2378b")];
#[crate_type = "lib"];

extern mod issue2378a;

use issue2378a::maybe;

pub struct two_maybes<T> {a: maybe<T>, b: maybe<T>}

impl<T:Clone> Index<uint,(T,T)> for two_maybes<T> {
    fn index(&self, idx: &uint) -> (T, T) {
        (self.a[*idx], self.b[*idx])
    }
}
