// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

static buf: &mut [u8] = &mut [1u8,2,3,4,5,7];   //[ast]~ ERROR E0017
                                                //[mir]~^ ERROR E0017
fn write<T: AsRef<[u8]>>(buffer: T) { }

fn main() {
    write(&buf);
    buf[0]=2;                                   //[ast]~ ERROR E0389
                                                //[mir]~^ ERROR E0594
}
