// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we correctly handle projection bounds appearing in the
// supertrait list (and in conjunction with overloaded operators). In
// this case, the `Result=Self` binding in the supertrait listing of
// `Int` was being ignored.

trait Not {
    type Result;

    fn not(self) -> Self::Result;
}

trait Int: Not<Result=Self> + Sized {
    fn count_ones(self) -> uint;
    fn count_zeros(self) -> uint {
        // neither works
        let x: Self = self.not();
        0
    }
}

fn main() { }
