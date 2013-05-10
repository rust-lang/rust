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
fn main(){
    let mut t1 = ~[];
    t1.push('a');

    let mut t2 = ~[];
    t2.push('b');

    for vec::each2_mut(t1, t2) | i1, i2 | {
        assert!(*i1 == 'a');
        assert!(*i2 == 'b');
    }

    for vec::each2(t1, t2) | i1, i2 | {
        io::println(fmt!("after t1: %?, t2: %?", i1, i2));
    }

    for vec::each2_mut(t1, t2) | i1, i2 | {
        *i1 = 'b';
        *i2 = 'a';
        assert!(*i1 == 'b');
        assert!(*i2 == 'a');
    }

    for vec::each2(t1, t2) | i1, i2 | {
        io::println(fmt!("before t1: %?, t2: %?", i1, i2));
    }
}
