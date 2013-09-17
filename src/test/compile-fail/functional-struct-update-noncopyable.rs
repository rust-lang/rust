// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// issue 7327

// xfail-fast #7103
extern mod extra;
use extra::arc::*;

struct A { y: Arc<int>, x: Arc<int> }

impl Drop for A {
    fn drop(&mut self) { println(fmt!("x=%?", self.x.get())); }
}
fn main() {
    let a = A { y: Arc::new(1), x: Arc::new(2) };
    let _b = A { y: Arc::new(3), ..a };
    let _c = a; //~ ERROR use of moved value
}
