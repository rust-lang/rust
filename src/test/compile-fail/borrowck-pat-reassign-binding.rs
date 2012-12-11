// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-pretty -- comments are infaithfully preserved

fn main() {
    let mut x: Option<int> = None;
    match x { //~ NOTE loan of mutable local variable granted here
      None => {}
      Some(ref i) => {
        // Not ok: i is an outstanding ptr into x.
        x = Some(*i+1); //~ ERROR assigning to mutable local variable prohibited due to outstanding loan
      }
    }
    copy x; // just to prevent liveness warnings
}
