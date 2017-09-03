// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that unsafe impl for Sync/Send can be provided for extern types.

#![feature(extern_types)]

extern {
    type A;
}

unsafe impl Sync for A { }
unsafe impl Send for A { }

fn assert_sync<T: ?Sized + Sync>() { }
fn assert_send<T: ?Sized + Send>() { }

fn main() {
    assert_sync::<A>();
    assert_send::<A>();
}
