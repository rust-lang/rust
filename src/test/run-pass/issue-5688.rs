// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
# Corrupted initialization in the static struct

...should print &[1, 2, 3] but instead prints something like
&[4492532864, 24]. It is pretty evident that the compiler messed up
with the representation of [int, ..n] and [int] somehow, or at least
failed to typecheck correctly.
*/

struct X { vec: &'static [int] }
static V: &'static [X] = &[X { vec: &[1, 2, 3] }];
pub fn main() {
    for &v in V.iter() {
        println(fmt!("%?", v.vec));
    }
}
