// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use std::panic;

pub struct DropLogger<'a> {
    id: usize,
    log: &'a panic::AssertUnwindSafe<RefCell<Vec<usize>>>
}

impl<'a> Drop for DropLogger<'a> {
    fn drop(&mut self) {
        self.log.0.borrow_mut().push(self.id);
    }
}

struct InjectedFailure;

#[allow(unreachable_code)]
fn main() {
    let log = panic::AssertUnwindSafe(RefCell::new(vec![]));
    let d = |id| DropLogger { id: id, log: &log };
    let get = || -> Vec<_> {
        let mut m = log.0.borrow_mut();
        let n = m.drain(..);
        n.collect()
    };

    {
        let _x = (d(0), &d(1), d(2), &d(3));
        // all borrows are extended - nothing has been dropped yet
        assert_eq!(get(), vec![]);
    }
    // in a let-statement, extended lvalues are dropped
    // *after* the let result (tho they have the same scope
    // as far as scope-based borrowck goes).
    assert_eq!(get(), vec![0, 2, 3, 1]);

    let _ = std::panic::catch_unwind(|| {
        (d(4), &d(5), d(6), &d(7), panic!(InjectedFailure));
    });

    // here, the temporaries (5/7) live until the end of the
    // containing statement, which is destroyed after the operands
    // (4/6) on a panic.
    assert_eq!(get(), vec![6, 4, 7, 5]);
}
