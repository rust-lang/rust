// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #33344, initial version. This example allowed
// arbitrary trait bounds to be synthesized.

trait Tweedledum: IntoIterator {}
trait Tweedledee: IntoIterator {}

impl<T: Tweedledum> Tweedledee for T {}
impl<T: Tweedledee> Tweedledum for T {}

trait Combo: IntoIterator {}
impl<T: Tweedledee + Tweedledum> Combo for T {}

fn is_ee<T: Combo>(t: T) {
    t.into_iter();
}

fn main() {
    is_ee(4);
    //~^ ERROR overflow evaluating the requirement `{integer}: Tweedle
}
