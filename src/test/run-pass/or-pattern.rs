// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum blah { a(int, int, uint), b(int, int), c, }

fn or_alt(q: blah) -> int {
    match q { a(x, y, _) | b(x, y) => { return x + y; } c => { return 0; } }
}

pub fn main() {
    assert!((or_alt(c) == 0));
    assert!((or_alt(a(10, 100, 0u)) == 110));
    assert!((or_alt(b(20, 200)) == 220));
}
