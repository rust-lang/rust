// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod std;
use comm::*;
use task::spawn;

fn a() {
    fn doit() {
        fn b(c: Chan<Chan<int>>) {
            let p = Port();
            send(c, Chan(&p));
        }
        let p = Port();
        let ch = Chan(&p);
        spawn(|| b(ch) );
        recv(p);
    }
    let mut i = 0;
    while i < 100 {
        doit();
        i += 1;
    }
}

fn main() {
    for iter::repeat(100u) {
        spawn(|| a() );
    }
}
