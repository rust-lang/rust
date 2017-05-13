// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

use std::collections::HashMap;

trait Map<K, V>
{
    fn get(&self, k: K) -> V { panic!() }
}

impl<K, V> Map<K, V> for HashMap<K, V> {}

// Test that trait types printed in error msgs include the type arguments.

fn main() {
    let x: Box<HashMap<isize, isize>> = box HashMap::new();
    let x: Box<Map<isize, isize>> = x;
    // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
    let y: Box<Map<usize, isize>> = Box::new(x);
    //~^ ERROR `Box<Map<isize, isize>>: Map<usize, isize>` is not satisfied
}
