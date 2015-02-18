// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

trait Get<T> : 'static {
    fn get(&self) -> T;
}

fn get_min_from_max<'min, 'max>(v: Box<Get<&'max i32>>)
                                -> Box<Get<&'min i32>>
    where 'max : 'min
{
    v
}

fn get_max_from_min<'min, 'max, G>(v: Box<Get<&'min i32>>)
                                   -> Box<Get<&'max i32>>
    where 'max : 'min
{
    v //~ ERROR mismatched types
}

fn main() { }
