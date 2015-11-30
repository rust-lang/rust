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
    let _ = #[cfg(unset)] ();
    //~^ ERROR removing an expression is not supported in this position
    let _ = 1 + 2 + #[cfg(unset)] 3;
    //~^ ERROR removing an expression is not supported in this position
    let _ = [1, 2, 3][#[cfg(unset)] 1];
    //~^ ERROR removing an expression is not supported in this position
}
