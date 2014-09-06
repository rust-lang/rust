// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn match_vecs<'a, T>(l1: &'a [T], l2: &'a [T]) {
    match (l1, l2) {
        ([], []) => println!("both empty"),
        ([], [hd, tl..]) | ([hd, tl..], []) => println!("one empty"),
        //~^ ERROR: cannot move out of dereference
        //~^^ ERROR: cannot move out of dereference
        ([hd1, tl1..], [hd2, tl2..]) => println!("both nonempty"),
        //~^ ERROR: cannot move out of dereference
        //~^^ ERROR: cannot move out of dereference
    }
}

fn main() {}
