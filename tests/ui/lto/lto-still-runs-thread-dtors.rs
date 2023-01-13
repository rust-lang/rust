// run-pass
// compile-flags: -C lto
// no-prefer-dynamic
// ignore-emscripten no threads support
// revisions: mir thir
// [thir]compile-flags: -Zthir-unsafeck

use std::thread;

static mut HIT: usize = 0;

thread_local!(static A: Foo = Foo);

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        unsafe {
            HIT += 1;
        }
    }
}

fn main() {
    unsafe {
        assert_eq!(HIT, 0);
        thread::spawn(|| {
            assert_eq!(HIT, 0);
            A.with(|_| ());
            assert_eq!(HIT, 0);
        }).join().unwrap();
        assert_eq!(HIT, 1);
    }
}
