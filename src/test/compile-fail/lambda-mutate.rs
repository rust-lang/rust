// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure we can't write to upvars from @fns
fn main() {
    let i = 0;
    let ctr: @fn() -> int = || { i = i + 1; i };
    //~^ ERROR cannot assign
    error!(ctr());
    error!(ctr());
    error!(ctr());
    error!(ctr());
    error!(ctr());
    error!(i);
}
