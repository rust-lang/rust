// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that we link regions in mutable place ops correctly - issue #41774

struct Data(i32);

trait OhNo {
    fn oh_no(&mut self, other: &Vec<Data>) { loop {} }
}

impl OhNo for Data {}
impl OhNo for [Data] {}

fn main() {
    let mut v = vec![Data(0)];
    v[0].oh_no(&v); //~ ERROR cannot borrow `v` as immutable because
    (*v).oh_no(&v); //~ ERROR cannot borrow `v` as immutable because
}
