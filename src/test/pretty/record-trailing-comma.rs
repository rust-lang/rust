// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test
// pp-exact
struct Thing {
    x: int,
    y: int,
}

fn main() {
    let sth = Thing{x: 0, y: 1,};
    let sth2 = Thing{y: 9 , ..sth};
    assert_eq!(sth.x + sth2.y, 9);
}
