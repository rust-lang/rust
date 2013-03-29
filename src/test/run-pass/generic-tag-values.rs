// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




// -*- rust -*-
enum noption<T> { some(T), }

struct Pair { x: int, y: int }

pub fn main() {
    let nop: noption<int> = some::<int>(5);
    match nop { some::<int>(n) => { debug!(n); assert!((n == 5)); } }
    let nop2: noption<Pair> = some(Pair{x: 17, y: 42});
    match nop2 {
      some(t) => {
        debug!(t.x);
        debug!(t.y);
        assert!((t.x == 17));
        assert!((t.y == 42));
      }
    }
}
