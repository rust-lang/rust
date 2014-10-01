// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A model for how the `Fn` traits could work. You can implement at
// most one of `Go`, `GoMut`, or `GoOnce`, and then the others follow
// automatically.

// aux-build:go_trait.rs

extern crate go_trait;

use go_trait::{Go, GoMut, GoOnce, go, go_mut, go_once};

use std::rc::Rc;
use std::cell::Cell;

///////////////////////////////////////////////////////////////////////////

struct SomeGoableThing {
    counter: Rc<Cell<int>>
}

impl Go for SomeGoableThing {
    fn go(&self, arg: int) {
        self.counter.set(self.counter.get() + arg);
    }
}

///////////////////////////////////////////////////////////////////////////

struct SomeGoOnceableThing {
    counter: Rc<Cell<int>>
}

impl GoOnce for SomeGoOnceableThing {
    fn go_once(self, arg: int) {
        self.counter.set(self.counter.get() + arg);
    }
}

///////////////////////////////////////////////////////////////////////////

fn main() {
    let counter = Rc::new(Cell::new(0));
    let mut x = SomeGoableThing { counter: counter.clone() };

    go(&x, 10);
    assert_eq!(counter.get(), 10);

    go_mut(&mut x, 100);
    assert_eq!(counter.get(), 110);

    go_once(x, 1_000);
    assert_eq!(counter.get(), 1_110);

    let x = SomeGoOnceableThing { counter: counter.clone() };

    go_once(x, 10_000);
    assert_eq!(counter.get(), 11_110);
}
