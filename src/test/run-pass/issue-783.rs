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
        fn b(c: core::oldcomm::Chan<core::oldcomm::Chan<int>>) {
            let p = core::oldcomm::Port();
            core::oldcomm::send(c, core::oldcomm::Chan(&p));
        }
        let p = core::oldcomm::Port();
        let ch = core::oldcomm::Chan(&p);
        task::spawn(|| b(ch) );
        core::oldcomm::recv(p);
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
