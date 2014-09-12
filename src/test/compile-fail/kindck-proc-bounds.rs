// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn is_send<T: Send>() {}
fn is_freeze<T: Sync>() {}

fn foo<'a>() {
    is_send::<proc()>();
    //~^ ERROR: the trait `core::kinds::Send` is not implemented

    is_freeze::<proc()>();
    //~^ ERROR: the trait `core::kinds::Sync` is not implemented
}

fn main() { }
