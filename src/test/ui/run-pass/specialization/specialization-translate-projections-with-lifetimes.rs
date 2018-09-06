// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

trait Iterator {
    fn next(&self);
}

trait WithAssoc {
    type Item;
}

impl<'a> WithAssoc for &'a () {
    type Item = &'a u32;
}

struct Cloned<I>(I);

impl<'a, I, T: 'a> Iterator for Cloned<I>
    where I: WithAssoc<Item=&'a T>, T: Clone
{
    fn next(&self) {}
}

impl<'a, I, T: 'a> Iterator for Cloned<I>
    where I: WithAssoc<Item=&'a T>, T: Copy
{

}

fn main() {
    Cloned(&()).next();
}
