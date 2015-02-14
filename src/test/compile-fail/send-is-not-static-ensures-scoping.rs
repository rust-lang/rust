// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core, std_misc)]
use std::thread::Thread;

fn main() {
    let bad = {
        let x = 1;
        let y = &x;

        Thread::scoped(|| { //~ ERROR cannot infer an appropriate lifetime
            let _z = y;
        })
    };

    bad.join().ok().unwrap();
}
