// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only -Z continue-parse-after-error

fn main() {
    let _: Box<((Copy)) + Copy>;
    //~^ ERROR expected a path on the left-hand side of `+`, not `((Copy))`
    let _: Box<(Copy + Copy) + Copy>;
    //~^ ERROR expected a path on the left-hand side of `+`, not `(Copy + Copy)`
    let _: Box<(Copy +) + Copy>;
    //~^ ERROR expected a path on the left-hand side of `+`, not `(Copy)`
    let _: Box<(dyn Copy) + Copy>;
    //~^ ERROR expected a path on the left-hand side of `+`, not `(dyn Copy)`
}
