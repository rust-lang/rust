// Copyright 2014–2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

// this tests the `unknown_lint` lint, especially the suggestions

// the suggestion only appears if a lint with the lowercase name exists
#[allow(FOO_BAR)]
// the suggestion appears on all-uppercase names
#[warn(DEAD_CODE)]
// the suggestion appears also on mixed-case names
#[deny(Warnings)]
fn main() {
    unimplemented!();
}
