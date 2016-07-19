// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let _ = Iterator::next(&mut ());
    //~^ ERROR `(): std::iter::Iterator` is not satisfied

    for _ in false {}
    //~^ ERROR `bool: std::iter::Iterator` is not satisfied

    let _ = Iterator::next(&mut ());
    //~^ ERROR `(): std::iter::Iterator` is not satisfied

    other()
}

pub fn other() {
    // check errors are still reported globally

    let _ = Iterator::next(&mut ());
    //~^ ERROR `(): std::iter::Iterator` is not satisfied

    let _ = Iterator::next(&mut ());
    //~^ ERROR `(): std::iter::Iterator` is not satisfied

    for _ in false {}
    //~^ ERROR `bool: std::iter::Iterator` is not satisfied
}
