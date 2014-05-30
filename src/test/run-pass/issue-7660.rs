// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regresion test for issue 7660
// rvalue lifetime too short when equivalent `match` works

extern crate collections;

use std::collections::HashMap;

struct A(int, int);

pub fn main() {
    let mut m: HashMap<int, A> = HashMap::new();
    m.insert(1, A(0, 0));

    let A(ref _a, ref _b) = *m.get(&1);
    let (a, b) = match *m.get(&1) { A(ref _a, ref _b) => (_a, _b) };
}
