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
    match nop { some::<int>(n) => { log(debug, n); fail_unless!((n == 5)); } }
    let nop2: noption<Pair> = some(Pair{x: 17, y: 42});
    match nop2 {
      some(t) => {
        log(debug, t.x);
        log(debug, t.y);
        fail_unless!((t.x == 17));
        fail_unless!((t.y == 42));
      }
    }
}
