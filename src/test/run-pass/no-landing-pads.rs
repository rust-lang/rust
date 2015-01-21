// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z no-landing-pads

use std::thread::Thread;

static mut HIT: bool = false;

struct A;

impl Drop for A {
    fn drop(&mut self) {
        unsafe { HIT = true; }
    }
}

fn main() {
    Thread::scoped(move|| -> () {
        let _a = A;
        panic!();
    }).join().unwrap_err();
    assert!(unsafe { !HIT });
}
