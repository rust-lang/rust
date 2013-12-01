// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static mut drop_count: uint = 0;

#[unsafe_no_drop_flag]
struct Foo {
    dropped: bool
}

impl Drop for Foo {
    fn drop(&mut self) {
        // Test to make sure we haven't dropped already
        assert!(!self.dropped);
        self.dropped = true;
        // And record the fact that we dropped for verification later
        unsafe { drop_count += 1; }
    }
}

pub fn main() {
    // An `if true { expr }` statement should compile the same as `{ expr }`.
    if true {
        let _a = Foo{ dropped: false };
    }
    // Check that we dropped already (as expected from a `{ expr }`).
    unsafe { assert!(drop_count == 1); }

    // An `if false {} else { expr }` statement should compile the same as `{ expr }`.
    if false {
        fail!();
    } else {
        let _a = Foo{ dropped: false };
    }
    // Check that we dropped already (as expected from a `{ expr }`).
    unsafe { assert!(drop_count == 2); }
}
