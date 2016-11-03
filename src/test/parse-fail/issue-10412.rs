// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only -Z continue-parse-after-error


trait Serializable<'self, T> { //~ ERROR lifetimes cannot use keyword names
    fn serialize(val : &'self T) -> Vec<u8> ; //~ ERROR lifetimes cannot use keyword names
    fn deserialize(repr : &[u8]) -> &'self T; //~ ERROR lifetimes cannot use keyword names
}

impl<'self> Serializable<str> for &'self str { //~ ERROR lifetimes cannot use keyword names
    //~^ ERROR lifetimes cannot use keyword names
    fn serialize(val : &'self str) -> Vec<u8> { //~ ERROR lifetimes cannot use keyword names
        vec![1]
    }
    fn deserialize(repr: &[u8]) -> &'self str { //~ ERROR lifetimes cannot use keyword names
        "hi"
    }
}

fn main() {
    println!("hello");
    let x = "foo".to_string();
    let y = x;
    println!("{}", y);
}
