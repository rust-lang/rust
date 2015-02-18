// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait T0 {
    type O;
    fn dummy(&self) { }
}

struct S<A>(A);
impl<A> T0 for S<A> { type O = A; }

trait T1: T0 {
    // this looks okay but as we see below, `f` is unusable
    fn m0<F: Fn(<Self as T0>::O) -> bool>(self, f: F) -> bool;
}

// complains about mismatched types: <S<A> as T0>::O vs. A
impl<A> T1 for S<A>
{
    fn m0<F: Fn(<Self as T0>::O) -> bool>(self, f: F) -> bool { f(self.0) }
}

fn main() { }
