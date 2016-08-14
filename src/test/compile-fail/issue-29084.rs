// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! foo {
    ($d:expr) => {{
        fn bar(d: u8) { }
        bar(&mut $d);
        //~^ ERROR mismatched types
        //~| expected u8, found &mut u8
        //~| expected type `u8`
        //~| found type `&mut u8`
    }}
}

fn main() {
    foo!(0u8);
    //~^ in this expansion of foo!
}
