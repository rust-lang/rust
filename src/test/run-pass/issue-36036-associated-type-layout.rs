// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue 36036: computing the layout of a type composed from another
// trait's associated type caused compiler to ICE when the associated
// type was allowed to be unsized, even though the known instantiated
// type is itself sized.

#![allow(dead_code)]

trait Context {
    type Container: ?Sized;
}

impl Context for u16 {
    type Container = u8;
}

struct Wrapper<C: Context+'static> {
    container: &'static C::Container
}

fn foobar(_: Wrapper<u16>) {}

static VALUE: u8 = 0;

fn main() {
    foobar(Wrapper { container: &VALUE });
}
