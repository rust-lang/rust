// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused, clippy::trivially_copy_pass_by_ref)]
#![warn(clippy::mut_from_ref)]

struct Foo;

impl Foo {
    fn this_wont_hurt_a_bit(&self) -> &mut Foo {
        unimplemented!()
    }
}

trait Ouch {
    fn ouch(x: &Foo) -> &mut Foo;
}

impl Ouch for Foo {
    fn ouch(x: &Foo) -> &mut Foo {
        unimplemented!()
    }
}

fn fail(x: &u32) -> &mut u16 {
    unimplemented!()
}

fn fail_lifetime<'a>(x: &'a u32, y: &mut u32) -> &'a mut u32 {
    unimplemented!()
}

fn fail_double<'a, 'b>(x: &'a u32, y: &'a u32, z: &'b mut u32) -> &'a mut u32 {
    unimplemented!()
}

// this is OK, because the result borrows y
fn works<'a>(x: &u32, y: &'a mut u32) -> &'a mut u32 {
    unimplemented!()
}

// this is also OK, because the result could borrow y
fn also_works<'a>(x: &'a u32, y: &'a mut u32) -> &'a mut u32 {
    unimplemented!()
}

fn main() {
    //TODO
}
