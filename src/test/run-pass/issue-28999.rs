// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct Xyz<'a, V> {
    pub v: (V, &'a u32),
}

pub fn eq<'a, 's, 't, V>(this: &'s Xyz<'a, V>, other: &'t Xyz<'a, V>) -> bool
        where V: PartialEq {
    this.v == other.v
}

fn main() {}
