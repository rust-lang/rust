// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Params;

pub trait Plugin<E: ?Sized> {
    type Error;
}

pub trait Pluggable {
    fn get_ref<P: Plugin<Self>>(&mut self) -> Option<P::Error> {
        None
    }
}

struct Foo;
impl Plugin<Foo> for Params {
    type Error = ();
}

impl<T: Copy> Pluggable for T {}

fn handle(req: &mut i32) {
    req.get_ref::<Params>();
    //~^ ERROR the trait bound `Params: Plugin<i32>` is not satisfied
}

fn main() {}
