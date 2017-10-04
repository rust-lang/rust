// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure extern types are !Sync and !Send.

#![feature(dynsized, extern_types)]

use std::marker::DynSized;

extern {
    type A;
}

fn assert_sync<T: ?DynSized + Sync>() { }
fn assert_send<T: ?DynSized + Send>() { }

fn main() {
    assert_sync::<A>();
    //~^ ERROR the trait bound `A: std::marker::Sync` is not satisfied

    assert_send::<A>();
    //~^ ERROR the trait bound `A: std::marker::Send` is not satisfied
}
