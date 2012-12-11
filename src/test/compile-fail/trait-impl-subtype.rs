// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Mumbo {
    fn jumbo(&self, x: @uint) -> uint;
}

impl uint: Mumbo {
    // Note: this method def is ok, it is more accepting and
    // less effecting than the trait method:
    pure fn jumbo(&self, x: @const uint) -> uint { *self + *x }
}

fn main() {
    let a = 3u;
    let b = a.jumbo(@mut 6);

    let x = @a as @Mumbo;
    let y = x.jumbo(@mut 6); //~ ERROR values differ in mutability
    let z = x.jumbo(@6);
}



