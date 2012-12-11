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

// We're trying to trigger a race between send and port destruction that
// results in the string not being freed

fn starship(&&ch: comm::Chan<~str>) {
    for int::range(0, 10) |_i| {
        comm::send(ch, ~"pew pew");
    }
}

fn starbase() {
    for int::range(0, 10) |_i| {
        let p = comm::Port();
        let c = comm::Chan(&p);
        task::spawn(|| starship(c) );
        task::yield();
    }
}

fn main() {
    for int::range(0, 10) |_i| {
        task::spawn(|| starbase() );
    }
}