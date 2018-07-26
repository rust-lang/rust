// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that unknown attribute error is shown even if there are unresolved macros.

#[marco_use] // typo
//~^ ERROR The attribute `marco_use` is currently unknown to the compiler
mod foo {
    macro_rules! bar {
        () => ();
    }
}

fn main() {
   bar!();
   //~^ ERROR cannot find macro `bar!`
}
