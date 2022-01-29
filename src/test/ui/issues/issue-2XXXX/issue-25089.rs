// run-pass
// needs-unwind
// ignore-emscripten no threads support

use std::thread;

struct Foo(i32);

impl Drop for Foo {
    fn drop(&mut self) {
        static mut DROPPED: bool = false;
        unsafe {
            assert!(!DROPPED);
            DROPPED = true;
        }
    }
}

struct Empty;

fn empty() -> Empty { Empty }

fn should_panic(_: Foo, _: Empty) {
    panic!("test panic");
}

fn test() {
    should_panic(Foo(1), empty());
}

fn main() {
    let ret = thread::spawn(test).join();
    assert!(ret.is_err());
}
