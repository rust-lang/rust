// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that impl methods are matched to traits exactly:
// we might be tempted to think matching is contravariant, but if
// we let an impl method can have more permissive bounds than the trait
// method it's implementing, the return type might be less specific than
// needed. Just punt and make it invariant.
use iter::BaseIter;

trait A {
  fn b<C:Copy Const, D>(x: C) -> C;
}

struct E {
 f: int
}

impl E: A {
  fn b<F:Copy, G>(_x: F) -> F { fail } //~ ERROR in method `b`, type parameter 0 has 1 bound, but
}

fn main() {}