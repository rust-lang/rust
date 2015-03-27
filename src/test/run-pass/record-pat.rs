// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

enum t1 { a(isize), b(usize), }
struct T2 {x: t1, y: isize}
enum t3 { c(T2, usize), }

fn m(input: t3) -> isize {
    match input {
      t3::c(T2 {x: t1::a(m), ..}, _) => { return m; }
      t3::c(T2 {x: t1::b(m), y: y}, z) => { return ((m + z) as isize) + y; }
    }
}

pub fn main() {
    assert_eq!(m(t3::c(T2 {x: t1::a(10), y: 5}, 4)), 10);
    assert_eq!(m(t3::c(T2 {x: t1::b(10), y: 5}, 4)), 19);
}
