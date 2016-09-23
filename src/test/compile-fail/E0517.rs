// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[repr(C)] //~ ERROR E0517
           //~| requires a struct, enum or union
type Foo = u8;

#[repr(packed)] //~ ERROR E0517
                //~| requires a struct
enum Foo2 {Bar, Baz}

#[repr(u8)] //~ ERROR E0517
            //~| requires an enum
struct Foo3 {bar: bool, baz: bool}

#[repr(C)] //~ ERROR E0517
           //~| requires a struct, enum or union
impl Foo3 {
}

fn main() {
}
