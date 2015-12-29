// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #14660

macro_rules! priv_x { () => {
    static x: u32 = 0;
}}

macro_rules! pub_x { () => {
    pub priv_x!(); //~ ERROR can't qualify macro invocation with `pub`
    //~^ HELP try adjusting the macro to put `pub` inside the invocation
}}

mod foo {
    pub_x!();
}

fn main() {
    let y: u32 = foo::x;
}
