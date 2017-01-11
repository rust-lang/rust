// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-allow_internal_unstable

macro_rules! bar {
    () => {
        // more layers don't help:
        #[allow_internal_unstable] //~ ERROR allow_internal_unstable side-steps
        macro_rules! baz {
            () => {}
        }
    }
}

bar!();

fn main() {}
