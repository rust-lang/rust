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

use std::cell::Cell;
use std::fmt;

struct Foo(Cell<int>);

impl fmt::Default for Foo {
    fn fmt(f: &Foo, _fmt: &mut fmt::Formatter) {
        let Foo(ref f) = *f;
        assert!(f.get() == 0);
        f.set(1);
    }
}

pub fn main() {
    let (p,c) = Chan::new();
    spawn(proc() {
        let mut f = Foo(Cell::new(0));
        debug!("{}", f);
        let Foo(ref mut f) = f;
        assert!(f.get() == 1);
        c.send(());
    });
    p.recv();
}
