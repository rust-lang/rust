// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use std::mem;

struct Test { f: usize }
impl Drop for Test {
    fn drop(&mut self) {}
}

fn main() {
    let mut x = (Test { f: 2 }, Test { f: 4 });
    mem::drop(x.0);
    x.0.f = 3;
    //~^ ERROR partial reinitialization of uninitialized structure `x.0`
}
