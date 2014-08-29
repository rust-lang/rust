// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn assert_send<T:Send>() { }

// unsafe ptrs are ok unless they point at unsendable things
fn test70() {
    assert_send::<*mut int>();
}
fn test71<'a>() {
    assert_send::<*mut &'a int>(); //~ ERROR does not fulfill the required lifetime
}

fn main() {
}
