// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Resources can't be copied, but storing into data structures counts
// as a move unless the stored thing is used afterwards.

struct r {
  i: @mut int,
}

struct Box { x: r }

#[unsafe_destructor]
impl Drop for r {
    fn drop(&self) {
        unsafe {
            *(self.i) = *(self.i) + 1;
        }
    }
}

fn r(i: @mut int) -> r {
    r {
        i: i
    }
}

fn test_box() {
    let i = @mut 0;
    {
        let a = @r(i);
    }
    assert_eq!(*i, 1);
}

fn test_rec() {
    let i = @mut 0;
    {
        let a = Box {x: r(i)};
    }
    assert_eq!(*i, 1);
}

fn test_tag() {
    enum t {
        t0(r),
    }

    let i = @mut 0;
    {
        let a = t0(r(i));
    }
    assert_eq!(*i, 1);
}

fn test_tup() {
    let i = @mut 0;
    {
        let a = (r(i), 0);
    }
    assert_eq!(*i, 1);
}

fn test_unique() {
    let i = @mut 0;
    {
        let a = ~r(i);
    }
    assert_eq!(*i, 1);
}

fn test_box_rec() {
    let i = @mut 0;
    {
        let a = @Box {
            x: r(i)
        };
    }
    assert_eq!(*i, 1);
}

pub fn main() {
    test_box();
    test_rec();
    test_tag();
    test_tup();
    test_unique();
    test_box_rec();
}
