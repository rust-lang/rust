// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Mirror<Smoke> {
    type Image;
}

impl<T, Smoke> Mirror<Smoke> for T {
    type Image = T;
}

pub fn poison<S>(victim: String) where <String as Mirror<S>>::Image: Copy {
    loop { drop(victim); }
}

fn main() {
    let s = "Hello!".to_owned();
    let mut s_copy = s;
    s_copy.push_str("World!");
    "0wned!".to_owned();
    println!("{}", s); //~ ERROR use of moved value
}
