// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that trait lifetime params aren't constrained to static lifetime
// when coercing something to owned trait

trait A<'a> {}
impl<'a> A<'a> for &'a A<'a> {}

struct B;
impl<'a> A<'a> for B {}
impl<'a> A<'a> for &'a B {}

pub fn main() {
    let bb = B;
    let _tmp = {
        let pb = ~&bb;
        let aa: ~A: = pb;
        aa
    };
}
