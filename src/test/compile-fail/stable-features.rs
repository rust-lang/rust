// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing that the stable_features lint catches use of stable
// language and lib features.

#![deny(stable_features)]
#![feature(test_accepted_feature)] //~ ERROR this feature has been stable since 1.0.0
#![feature(rust1)] //~ ERROR this feature has been stable since 1.0.0

fn main() {
    let _foo: Vec<()> = Vec::new();
}
