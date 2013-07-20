// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests the passing down of expected types through boxing and
// wrapping in a record or tuple. (The a.x would complain about 'this
// type must be known in this context' if the passing down doesn't
// happen.)

fn eat_tup(_r: ~@(int, @fn(Pair) -> int)) {}
fn eat_rec(_r: ~Rec) {}

struct Rec<'self> { a: int, b: &'self fn(Pair) -> int }
struct Pair { x: int, y: int }

pub fn main() {
    eat_tup(~@(10, |a| a.x ));
    eat_rec(~Rec{a: 10, b: |a| a.x });
}
