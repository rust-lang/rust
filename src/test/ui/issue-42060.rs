// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let thing = ();
    let other: typeof(thing) = thing; //~ ERROR attempt to use a non-constant value in a constant
    //~^ ERROR `typeof` is a reserved keyword but unimplemented [E0516]
}

fn f(){
    let q = 1;
    <typeof(q)>::N //~ ERROR attempt to use a non-constant value in a constant
    //~^ ERROR `typeof` is a reserved keyword but unimplemented [E0516]
}

