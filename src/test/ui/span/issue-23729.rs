// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #23729

fn main() {
    let fib = {
        struct Recurrence {
            mem: [u64; 2],
            pos: usize,
        }

        impl Iterator for Recurrence {
            //~^ ERROR E0046
            //~| NOTE missing `Item` in implementation
            //~| NOTE `Item` from trait: `type Item;`
            #[inline]
            fn next(&mut self) -> Option<u64> {
                if self.pos < 2 {
                    let next_val = self.mem[self.pos];
                    self.pos += 1;
                    Some(next_val)
                } else {
                    let next_val = self.mem[0] + self.mem[1];
                    self.mem[0] = self.mem[1];
                    self.mem[1] = next_val;
                    Some(next_val)
                }
            }
        }

        Recurrence { mem: [0, 1], pos: 0 }
    };

    for e in fib.take(10) {
        println!("{}", e)
    }
}
