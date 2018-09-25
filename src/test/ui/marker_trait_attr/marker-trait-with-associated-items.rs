// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(marker_trait_attr)]
#![feature(associated_type_defaults)]

#[marker]
trait MarkerConst {
    const N: usize;
    //~^ ERROR marker traits cannot have associated items
}

#[marker]
trait MarkerType {
    type Output;
    //~^ ERROR marker traits cannot have associated items
}

#[marker]
trait MarkerFn {
    fn foo();
    //~^ ERROR marker traits cannot have associated items
}

#[marker]
trait MarkerConstWithDefault {
    const N: usize = 43;
    //~^ ERROR marker traits cannot have associated items
}

#[marker]
trait MarkerTypeWithDefault {
    type Output = ();
    //~^ ERROR marker traits cannot have associated items
}

#[marker]
trait MarkerFnWithDefault {
    fn foo() {}
    //~^ ERROR marker traits cannot have associated items
}

fn main() {}
