// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test sized-ness checking in substitution in impls.

// impl - struct
trait T3<Z: ?Sized> {
    fn foo(&self, z: &Z);
}

struct S5<Y>(Y);

impl<X: ?Sized> T3<X> for S5<X> { //~ ERROR not implemented
}

fn main() { }
