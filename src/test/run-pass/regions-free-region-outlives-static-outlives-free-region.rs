// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we recognize that if you have
//
//     'a : 'static
//
// then
//
//     'a : 'b

fn test<'a,'b>(x: &'a i32) -> &'b i32
    where 'a: 'static
{
    x
}

fn main() { }
