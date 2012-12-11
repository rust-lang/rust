// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #763

extern mod std;
use comm::Chan;
use comm::send;
use comm::Port;
use comm::recv;

enum request { quit, close(Chan<bool>), }

type ctx = Chan<request>;

fn request_task(c: Chan<ctx>) {
    let p = Port();
    send(c, Chan(&p));
    let mut req: request;
    req = recv(p);
    // Need to drop req before receiving it again
    req = recv(p);
}

fn new_cx() -> ctx {
    let p = Port();
    let ch = Chan(&p);
    let t = task::spawn(|| request_task(ch) );
    let mut cx: ctx;
    cx = recv(p);
    return cx;
}

fn main() {
    let cx = new_cx();

    let p = Port::<bool>();
    send(cx, close(Chan(&p)));
    send(cx, quit);
}
