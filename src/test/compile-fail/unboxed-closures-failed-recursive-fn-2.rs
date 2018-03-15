// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Various unsuccessful attempts to put the unboxed closure kind
// inference into an awkward position that might require fixed point
// iteration (basically where inferring the kind of a closure `c`
// would require knowing the kind of `c`). I currently believe this is
// impossible.

fn a() {
    let mut closure0 = None;
    let vec = vec![1, 2, 3];

    loop {
        {
            let closure1 = || {
                match closure0.take() {
                    Some(c) => {
                        return c();
                        //~^ ERROR the type of this value must be known in this context
                    }
                    None => { }
                }
            };
            closure1();
        }

        closure0 = || vec;
    }
}

fn main() { }
