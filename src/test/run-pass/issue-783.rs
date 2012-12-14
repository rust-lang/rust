// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn a() {
    fn doit() {
        fn b(c: core::comm::Chan<core::comm::Chan<int>>) {
            let p = core::comm::Port();
            core::comm::send(c, core::comm::Chan(&p));
        }
        let p = core::comm::Port();
        let ch = core::comm::Chan(&p);
        task::spawn(|| b(ch) );
        core::comm::recv(p);
    }
    let mut i = 0;
    while i < 100 {
        doit();
        i += 1;
    }
}

fn main() {
    for iter::repeat(100u) {
        task::spawn(|| a() );
    }
}
