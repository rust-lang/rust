// run-pass
// compile-flags: -Z no-landing-pads -C codegen-units=1
// ignore-emscripten no threads support
// ignore-test fails because catch_unwind doesn't work with no-landing-pads

use std::thread;

static mut HIT: bool = false;

struct A;

impl Drop for A {
    fn drop(&mut self) {
        unsafe { HIT = true; }
    }
}

fn main() {
    thread::spawn(move|| -> () {
        let _a = A;
        panic!();
    }).join().unwrap_err();
    assert!(unsafe { !HIT });
}
