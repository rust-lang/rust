// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that derive on a packed struct with non-Copy fields
// correctly. This can't be made to work perfectly because
// we can't just use the field from the struct as it might
// not be aligned.

#[derive(PartialEq)]
struct Y(usize);

#[derive(PartialEq)]
//~^ ERROR cannot move out of borrowed
//~| ERROR cannot move out of borrowed
//~| ERROR cannot move out of borrowed
//~| ERROR cannot move out of borrowed
#[repr(packed)]
struct X(Y);

fn main() {
}
