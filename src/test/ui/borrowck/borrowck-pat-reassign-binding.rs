// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn main() {
    let mut x: Option<isize> = None;
    match x {
      None => {
          // Note: on this branch, no borrow has occurred.
          x = Some(0);
      }
      Some(ref i) => {
          // But on this branch, `i` is an outstanding borrow
          x = Some(*i+1); //[ast]~ ERROR cannot assign to `x`
          //[mir]~^ ERROR cannot assign to `x` because it is borrowed
          drop(i);
      }
    }
    x.clone(); // just to prevent liveness warnings
}
