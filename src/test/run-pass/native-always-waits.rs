// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast
// ignore-android (FIXME #11419)

extern mod native;

static mut set: bool = false;

#[start]
fn start(argc: int, argv: **u8) -> int {
    // make sure that native::start always waits for all children to finish
    native::start(argc, argv, proc() {
        spawn(proc() {
            unsafe { set = true; }
        });
    });

    // if we didn't set the global, then return a nonzero code
    if unsafe {set} {0} else {1}
}
