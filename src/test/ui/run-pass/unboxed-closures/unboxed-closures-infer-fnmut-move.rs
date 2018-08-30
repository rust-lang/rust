// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are able to infer a suitable kind for this `move`
// closure that is just called (`FnMut`).

fn main() {
    let mut counter = 0;

    let v = {
        let mut tick = move || { counter += 1; counter };
        tick();
        tick()
    };

    assert_eq!(counter, 0);
    assert_eq!(v, 2);
}
