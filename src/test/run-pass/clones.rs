// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let a : ~int = ~5i;
    let b : ~int = a.clone();

    debug!(fmt!("a: %?, b: %?", a, b));
    
    let a : @int = @5i;
    let b : @int = a.clone();

    debug!(fmt!("a: %?, b: %?", a, b));

    let a : @mut int = @mut 5i;
    let b : @mut int = a.clone();
    *b = 6;

    debug!(fmt!("a: %?, b: %?", a, b));
}
