// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn main() {
    #macro[[#apply[f, [x, ...]], f(x, ...)]];

    macro_rules! apply_tt(
        ($f:expr, ($($x:expr),*)) => {$f($($x),*)}
    )

    fn add(a: int, b: int) -> int { return a + b; }

    assert(apply!(add, [1, 15]) == 16);
    assert(apply!(add, [1, 15]) == 16);
    assert(apply_tt!(add, (1, 15)) == 16);
}
