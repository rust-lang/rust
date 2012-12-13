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

enum request { quit, close(core::comm::Chan<bool>), }

type ctx = core::comm::Chan<request>;

fn request_task(c: core::comm::Chan<ctx>) {
    let p = core::comm::Port();
    core::comm::send(c, core::comm::Chan(&p));
    let mut req: request;
    req = core::comm::recv(p);
    // Need to drop req before receiving it again
    req = core::comm::recv(p);
}

fn new_cx() -> ctx {
    let p = core::comm::Port();
    let ch = core::comm::Chan(&p);
    let t = task::spawn(|| request_task(ch) );
    let mut cx: ctx;
    cx = core::comm::recv(p);
    return cx;
}

fn main() {
    let cx = new_cx();

    let p = core::comm::Port::<bool>();
    core::comm::send(cx, close(core::comm::Chan(&p)));
    core::comm::send(cx, quit);
}
