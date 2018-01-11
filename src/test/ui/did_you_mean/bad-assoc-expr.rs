// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let a = [1, 2, 3, 4];
    [i32; 4]::clone(&a);
    //~^ ERROR missing angle brackets in associated item path

    [i32]::as_ref(&a);
    //~^ ERROR missing angle brackets in associated item path

    (u8)::clone(&0);
    //~^ ERROR missing angle brackets in associated item path

    (u8, u8)::clone(&(0, 0));
    //~^ ERROR missing angle brackets in associated item path

    &(u8)::clone(&0);
    //~^ ERROR missing angle brackets in associated item path

    10 + (u8)::clone(&0);
    //~^ ERROR missing angle brackets in associated item path
}
