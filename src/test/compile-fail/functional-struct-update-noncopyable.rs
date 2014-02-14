// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// issue 7327

// ignore-fast #7103
extern crate sync;
use sync::Arc;

struct A { y: Arc<int>, x: Arc<int> }

impl Drop for A {
    fn drop(&mut self) { println!("x={:?}", self.x.get()); }
}
fn main() {
    let a = A { y: Arc::new(1), x: Arc::new(2) };
    let _b = A { y: Arc::new(3), ..a }; //~ ERROR cannot move out of type `A`
    let _c = a;
}
