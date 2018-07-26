// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for hashing involving canonical variables.  In this
// test -- which has an intensional error -- the type of the value
// being dropped winds up including a type variable. Canonicalization
// would then produce a `?0` which -- in turn -- triggered an ICE in
// hashing.

// revisions:cfail1

fn main() {
    println!("Hello, world! {}",*thread_rng().choose(&[0, 1, 2, 3]).unwrap());
    //[cfail1]~^ ERROR cannot find function `thread_rng`
}
