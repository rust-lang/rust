// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that when we test the supertrait we ensure consistent use of
// lifetime parameters. In this case, implementing T2<'a,'b> requires
// an impl of T1<'a>, but we have an impl of T1<'b>.

trait T1<'x> {
    fn x(&self) -> &'x int;
}

trait T2<'x, 'y> : T1<'x> {
    fn y(&self) -> &'y int;
}

struct S<'a, 'b> {
    a: &'a int,
    b: &'b int
}

impl<'a,'b> T1<'b> for S<'a, 'b> {
    fn x(&self) -> &'b int {
        self.b
    }
}

impl<'a,'b> T2<'a, 'b> for S<'a, 'b> { //~ ERROR cannot infer an appropriate lifetime
    fn y(&self) -> &'b int {
        self.b
    }
}

pub fn main() {
}
