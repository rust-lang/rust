// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod Bar {
    pub struct Foo( bool, pub i32, f32, bool);
    //~^ NOTE private field declared here
    //~| NOTE private field declared here
    //~| NOTE private field declared here
}

fn main() {
    let f = Bar::Foo(false,1,0.1, true); //~ ERROR E0450
                         //~^ NOTE cannot construct with a private field
}
