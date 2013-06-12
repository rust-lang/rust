// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


extern mod extra;

fn foo<T>(y: Option<T>) {
    let mut x: int;
    let mut rs: ~[int] = ~[];
    /* tests that x doesn't get put in the precondition for the
       entire if expression */

    if true {
    } else {
        match y {
          None::<T> => x = 17,
          _ => x = 42
        }
        rs.push(x);
    }
    return;
}

pub fn main() { debug!("hello"); foo::<int>(Some::<int>(5)); }
