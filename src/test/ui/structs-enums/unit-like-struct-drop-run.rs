// run-pass
// needs-unwind
// ignore-emscripten no threads support

// Make sure the destructor is run for unit-like structs.

use std::thread;

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        panic!("This panic should happen.");
    }
}

pub fn main() {
    let x = thread::spawn(move|| {
        let _b = Foo;
    }).join();

    let s = x.unwrap_err().downcast::<&'static str>().unwrap();
    assert_eq!(&**s, "This panic should happen.");
}
