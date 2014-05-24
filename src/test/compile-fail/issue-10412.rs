// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait Serializable<'self, T> { //~ ERROR no longer a special lifetime
    fn serialize(val : &'self T) -> Vec<u8> ; //~ ERROR no longer a special lifetime
    fn deserialize(repr : &[u8]) -> &'self T; //~ ERROR no longer a special lifetime
}

impl<'self> Serializable<str> for &'self str { //~ ERROR no longer a special lifetime
    //~^ ERROR no longer a special lifetime
    fn serialize(val : &'self str) -> Vec<u8> { //~ ERROR no longer a special lifetime
        vec!(1)
    }
    fn deserialize(repr: &[u8]) -> &'self str { //~ ERROR no longer a special lifetime
        "hi"
    }
}

fn main() {
    println!("hello");
    let x = "foo".to_owned();
    let y = x;
    println!("{}", y);
}
