// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are able to infer a suitable kind for this closure
// that is just called (`FnMut`).

fn main() {
    let mut counter = 0;

    {
        let mut tick = || counter += 1;
        tick();
        tick();
    }

    assert_eq!(counter, 2);
}
