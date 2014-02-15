// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast

extern crate extra;

use std::task;

struct complainer {
  c: Chan<bool>,
}

impl Drop for complainer {
    fn drop(&mut self) {
        error!("About to send!");
        self.c.send(true);
        error!("Sent!");
    }
}

fn complainer(c: Chan<bool>) -> complainer {
    error!("Hello!");
    complainer {
        c: c
    }
}

fn f(c: Chan<bool>) {
    let _c = complainer(c);
    fail!();
}

pub fn main() {
    let (p, c) = Chan::new();
    task::spawn(proc() f(c.clone()));
    error!("hiiiiiiiii");
    assert!(p.recv());
}
