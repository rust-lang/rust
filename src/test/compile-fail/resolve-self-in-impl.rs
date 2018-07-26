// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_type_defaults)]

struct S<T = u8>(T);
trait Tr<T = u8> {
    type A = ();
}

impl Tr<Self> for S {} // OK
impl<T: Tr<Self>> Tr<T> for S {} // OK
impl Tr for S where Self: Copy {} // OK
impl Tr for S where S<Self>: Copy {} // OK
impl Tr for S where Self::A: Copy {} // OK

impl Tr for Self {} //~ ERROR cycle detected
impl Tr for S<Self> {} //~ ERROR cycle detected
impl Self {} //~ ERROR cycle detected
impl S<Self> {} //~ ERROR cycle detected
impl Tr<Self::A> for S {} //~ ERROR cycle detected

fn main() {}
