// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S<T = u8>(T);
trait Tr<T = u8> {}

impl Tr<Self> for S {} // OK

// FIXME: `Self` cannot be used in bounds because it depends on bounds itself.
impl<T: Tr<Self>> Tr<T> for S {} //~ ERROR `Self` type is used before it's determined
impl<T = Self> Tr<T> for S {} //~ ERROR `Self` type is used before it's determined
impl Tr for S where Self: Copy {} //~ ERROR `Self` type is used before it's determined
impl Tr for S where S<Self>: Copy {} //~ ERROR `Self` type is used before it's determined
impl Tr for S where Self::Assoc: Copy {} //~ ERROR `Self` type is used before it's determined
                                         //~^ ERROR `Self` type is used before it's determined
impl Tr for Self {} //~ ERROR `Self` type is used before it's determined
impl Tr for S<Self> {} //~ ERROR `Self` type is used before it's determined
impl Self {} //~ ERROR `Self` type is used before it's determined
impl S<Self> {} //~ ERROR `Self` type is used before it's determined
impl Tr<Self::Assoc> for S {} //~ ERROR `Self` type is used before it's determined

fn main() {}
