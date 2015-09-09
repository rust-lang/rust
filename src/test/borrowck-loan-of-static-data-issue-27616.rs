// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

fn leak<T>(mut b: Box<T>) -> &'static mut T {
    // isn't this supposed to be safe?
    let inner = &mut *b as *mut _;
    mem::forget(b);
    unsafe { &mut *inner }
}

fn evil(mut s: &'static mut String)
{
    // create alias
    let alias: &'static mut String = s;
    let inner: &str = &alias;
    // free value
    *s = String::new(); //~ ERROR cannot assign
    let _spray = "0wned".to_owned();
    // ... and then use it
    println!("{}", inner);
}

fn main() {
    evil(leak(Box::new("hello".to_owned())));
}
