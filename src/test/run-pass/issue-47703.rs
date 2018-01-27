// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

struct MyStruct<'a> {
    field: &'a mut (),
    field2: WithDrop
}

struct WithDrop;

impl Drop for WithDrop {
    fn drop(&mut self) {}
}

impl<'a> MyStruct<'a> {
    fn consume(self) -> &'a mut () { self.field }
}

fn main() {}
