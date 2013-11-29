// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
// exec-env:RUST_LOG=debug

#[feature(managed_boxes)];

use std::fmt;

struct Foo(@mut int);

impl fmt::Default for Foo {
    fn fmt(f: &Foo, _fmt: &mut fmt::Formatter) {
        assert!(***f == 0);
        ***f = 1;
    }
}

pub fn main() {
    let (p,c) = stream();
    do spawn {
        let f = Foo(@mut 0);
        debug!("{}", f);
        assert!(**f == 1);
        c.send(());
    }
    p.recv();
}
