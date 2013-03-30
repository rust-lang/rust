// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum t1 { a(int), b(uint), }
struct T2 {x: t1, y: int}
enum t3 { c(T2, uint), }

fn m(in: t3) -> int {
    match in {
      c(T2 {x: a(m), _}, _) => { return m; }
      c(T2 {x: b(m), y: y}, z) => { return ((m + z) as int) + y; }
    }
}

pub fn main() {
    assert!((m(c(T2 {x: a(10), y: 5}, 4u)) == 10));
    assert!((m(c(T2 {x: b(10u), y: 5}, 4u)) == 19));
}
