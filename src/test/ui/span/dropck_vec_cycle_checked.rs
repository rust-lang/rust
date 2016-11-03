// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Reject mixing cyclic structure and Drop when using Vec.
//
// (Compare against compile-fail/dropck_arr_cycle_checked.rs)

#![feature(const_fn)]

use std::cell::Cell;
use id::Id;

mod s {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static S_COUNT: AtomicUsize = AtomicUsize::new(0);

    pub fn next_count() -> usize {
        S_COUNT.fetch_add(1, Ordering::SeqCst) + 1
    }
}

mod id {
    use s;
    #[derive(Debug)]
    pub struct Id {
        orig_count: usize,
        count: usize,
    }

    impl Id {
        pub fn new() -> Id {
            let c = s::next_count();
            println!("building Id {}", c);
            Id { orig_count: c, count: c }
        }
        pub fn count(&self) -> usize {
            println!("Id::count on {} returns {}", self.orig_count, self.count);
            self.count
        }
    }

    impl Drop for Id {
        fn drop(&mut self) {
            println!("dropping Id {}", self.count);
            self.count = 0;
        }
    }
}

trait HasId {
    fn count(&self) -> usize;
}

#[derive(Debug)]
struct CheckId<T:HasId> {
    v: T
}

#[allow(non_snake_case)]
fn CheckId<T:HasId>(t: T) -> CheckId<T> { CheckId{ v: t } }

impl<T:HasId> Drop for CheckId<T> {
    fn drop(&mut self) {
        assert!(self.v.count() > 0);
    }
}

#[derive(Debug)]
struct C<'a> {
    id: Id,
    v: Vec<CheckId<Cell<Option<&'a C<'a>>>>>,
}

impl<'a> HasId for Cell<Option<&'a C<'a>>> {
    fn count(&self) -> usize {
        match self.get() {
            None => 1,
            Some(c) => c.id.count(),
        }
    }
}

impl<'a> C<'a> {
    fn new() -> C<'a> {
        C { id: Id::new(), v: Vec::new() }
    }
}

fn f() {
    let (mut c1, mut c2, mut c3);
    c1 = C::new();
    c2 = C::new();
    c3 = C::new();

    c1.v.push(CheckId(Cell::new(None)));
    c1.v.push(CheckId(Cell::new(None)));
    c2.v.push(CheckId(Cell::new(None)));
    c2.v.push(CheckId(Cell::new(None)));
    c3.v.push(CheckId(Cell::new(None)));
    c3.v.push(CheckId(Cell::new(None)));

    c1.v[0].v.set(Some(&c2));
    c1.v[1].v.set(Some(&c3));
    c2.v[0].v.set(Some(&c2));
    c2.v[1].v.set(Some(&c3));
    c3.v[0].v.set(Some(&c1));
    c3.v[1].v.set(Some(&c2));
}
//~^ ERROR `c2` does not live long enough
//~| ERROR `c3` does not live long enough
//~| ERROR `c2` does not live long enough
//~| ERROR `c3` does not live long enough
//~| ERROR `c1` does not live long enough
//~| ERROR `c2` does not live long enough

fn main() {
    f();
}
