// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate debug;

fn baz() -> ! { fail!(); }

fn foo() {
    match Some::<int>(5) {
      Some::<int>(_x) => {
        let mut bar;
        match None::<int> { None::<int> => { bar = 5i; } _ => { baz(); } }
        println!("{:?}", bar);
      }
      None::<int> => { println!("hello"); }
    }
}

pub fn main() { foo(); }
