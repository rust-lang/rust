// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub enum Attribute {
    Code {attr_name_idx: u16},
}

pub enum Foo {
    Bar
}

fn test(a: Foo) {
    println!("{}", a.b); //~ ERROR attempted access of field
}

fn main() {
    let x = Attribute::Code {
        attr_name_idx: 42,
    };
    let z = (&x).attr_name_idx; //~ ERROR attempted access of field
    let y = x.attr_name_idx; //~ ERROR attempted access of field
}
