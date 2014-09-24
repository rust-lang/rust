// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A battery of tests to ensure destructors of unboxed closure environments
// run at the right times.

#![feature(overloaded_calls, unboxed_closures)]

static mut DROP_COUNT: uint = 0;

fn drop_count() -> uint {
    unsafe {
        DROP_COUNT
    }
}

struct Droppable {
    x: int,
}

impl Droppable {
    fn new() -> Droppable {
        Droppable {
            x: 1
        }
    }
}

impl Drop for Droppable {
    fn drop(&mut self) {
        unsafe {
            DROP_COUNT += 1
        }
    }
}

fn a<F:Fn(int, int) -> int>(f: F) -> int {
    f(1, 2)
}

fn b<F:FnMut(int, int) -> int>(mut f: F) -> int {
    f(3, 4)
}

fn c<F:FnOnce(int, int) -> int>(f: F) -> int {
    f(5, 6)
}

fn test_fn() {
    {
        a(move |&: a: int, b| { a + b });
    }
    assert_eq!(drop_count(), 0);

    {
        let z = &Droppable::new();
        a(move |&: a: int, b| { z; a + b });
        assert_eq!(drop_count(), 0);
    }
    assert_eq!(drop_count(), 1);

    {
        let z = &Droppable::new();
        let zz = &Droppable::new();
        a(move |&: a: int, b| { z; zz; a + b });
        assert_eq!(drop_count(), 1);
    }
    assert_eq!(drop_count(), 3);
}

fn test_fn_mut() {
    {
        b(move |&mut: a: int, b| { a + b });
    }
    assert_eq!(drop_count(), 3);

    {
        let z = &Droppable::new();
        b(move |&mut: a: int, b| { z; a + b });
        assert_eq!(drop_count(), 3);
    }
    assert_eq!(drop_count(), 4);

    {
        let z = &Droppable::new();
        let zz = &Droppable::new();
        b(move |&mut: a: int, b| { z; zz; a + b });
        assert_eq!(drop_count(), 4);
    }
    assert_eq!(drop_count(), 6);
}

fn test_fn_once() {
    {
        c(move |: a: int, b| { a + b });
    }
    assert_eq!(drop_count(), 6);

    {
        let z = Droppable::new();
        c(move |: a: int, b| { z; a + b });
        assert_eq!(drop_count(), 7);
    }
    assert_eq!(drop_count(), 7);

    {
        let z = Droppable::new();
        let zz = Droppable::new();
        c(move |: a: int, b| { z; zz; a + b });
        assert_eq!(drop_count(), 9);
    }
    assert_eq!(drop_count(), 9);
}

fn main() {
    test_fn();
    test_fn_mut();
    test_fn_once();
}

