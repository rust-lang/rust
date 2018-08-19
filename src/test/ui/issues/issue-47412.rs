// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Copy, Clone)]
enum Void {}

// Tests that we detect unsafe places (specifically, union fields and
// raw pointer dereferences), even when they're matched on while having
// an uninhabited type (equivalent to `std::intrinsics::unreachable()`).

fn union_field() {
    union Union { unit: (), void: Void }
    let u = Union { unit: () };
    match u.void {}
    //~^ ERROR access to union field is unsafe
}

fn raw_ptr_deref() {
    let ptr = std::ptr::null::<Void>();
    match *ptr {}
    //~^ ERROR dereference of raw pointer is unsafe
}

fn main() {}
