// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-win32
extern mod std;

use core::comm::*;

struct complainer {
  c: SharedChan<bool>,
}

impl Drop for complainer {
    fn finalize(&self) {
        error!("About to send!");
        self.c.send(true);
        error!("Sent!");
    }
}

fn complainer(c: SharedChan<bool>) -> complainer {
    error!("Hello!");
    complainer {
        c: c
    }
}

fn f(c: SharedChan<bool>) {
    let _c = complainer(c);
    fail!();
}

pub fn main() {
    let (p, c) = stream();
    let c = SharedChan(c);
    task::spawn_unlinked(|| f(c.clone()) );
    error!("hiiiiiiiii");
    assert!(p.recv());
}
