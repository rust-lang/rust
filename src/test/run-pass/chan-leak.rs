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

enum request { quit, close(core::oldcomm::Chan<bool>), }

type ctx = core::oldcomm::Chan<request>;

fn request_task(c: core::oldcomm::Chan<ctx>) {
    let p = core::oldcomm::Port();
    core::oldcomm::send(c, core::oldcomm::Chan(&p));
    let mut req: request;
    req = core::oldcomm::recv(p);
    // Need to drop req before receiving it again
    req = core::oldcomm::recv(p);
}

fn new_cx() -> ctx {
    let p = core::oldcomm::Port();
    let ch = core::oldcomm::Chan(&p);
    let t = task::spawn(|| request_task(ch) );
    let mut cx: ctx;
    cx = core::oldcomm::recv(p);
    return cx;
}

fn main() {
    let cx = new_cx();

    let p = core::oldcomm::Port::<bool>();
    core::oldcomm::send(cx, close(core::oldcomm::Chan(&p)));
    core::oldcomm::send(cx, quit);
}
