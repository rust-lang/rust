// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(untagged_unions)]
#![feature(nll)]

use std::ptr;

trait Trait {}
impl<T> Trait for T {}

#[allow(unions_with_drop_fields)]
union NoDrop<T: ?Sized> {
    value: T,
}

struct ShouldDrop<'a> {
    dropped: &'a mut bool
}

impl<'a> Drop for ShouldDrop<'a> {
    fn drop(&mut self) {
        *self.dropped = true;
    }
}

struct ShouldntDrop;

impl Drop for ShouldntDrop {
    fn drop(&mut self) {
        unsafe {
            panic!("This should not be dropped!");
        }
    }
}

fn main() {
    let mut dropped = false;
    {
        let mut should_drop = &mut NoDrop {
            value: ShouldDrop {
                dropped: &mut dropped
            }
        } as &mut NoDrop<Trait>;

        unsafe {
            ptr::drop_in_place(&mut should_drop.value);
        }
    }

    assert!(dropped);

    // NoDrop will be dropped, but the ShouldntDrop won't be
    Box::new(NoDrop { value: ShouldntDrop }) as Box<NoDrop<Trait>>;

    // FIXME: do something with Bar
}
