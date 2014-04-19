// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that bounds are allowed on structs and enums.

trait T {}

struct S<X: T>;

enum E<X: T> {
}

impl<X: T> S<X> {}
impl<X: T> E<X> {}

trait T2<'a> {}

struct S2<'a, X: T2<'a>>;

fn main() {}
