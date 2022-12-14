// run-pass
// needs-unwind
// ignore-emscripten no threads support

use std::thread;

fn f() {
    let _a: Box<_> = Box::new(0);
    panic!();
}

pub fn main() {
    let t = thread::spawn(f);
    drop(t.join());
}
