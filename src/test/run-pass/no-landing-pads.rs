// compile-flags: -Z no-landing-pads -C codegen-units=1
// ignore-emscripten no threads support

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
