// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:meep
extern mod std;
use comm::Chan;
use comm::Port;
use comm::send;
use comm::recv;

fn echo<T: Send>(c: Chan<T>, oc: Chan<Chan<T>>) {
    // Tests that the type argument in port gets
    // visited
    let p = Port::<T>();
    send(oc, Chan(&p));

    let x = recv(p);
    send(c, move x);
}

fn main() { fail ~"meep"; }
