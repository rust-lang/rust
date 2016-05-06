// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name="a"]
#![crate_type = "lib"]

#![allow(unknown_features)]
#![feature(box_syntax)]

pub trait i<T>
{
    fn dummy(&self, t: T) -> T { panic!() }
}

pub fn f<T>() -> Box<i<T>+'static> {
    impl<T> i<T> for () { }

    box () as Box<i<T>+'static>
}
