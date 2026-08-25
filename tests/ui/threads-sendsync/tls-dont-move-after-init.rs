//@ run-pass
//@ needs-threads

use std::cell::Cell;
use std::thread;

#[derive(Default)]
struct Foo {
    ptr: Cell<*const Foo>,
}

impl Foo {
    fn touch(&self) {
        if self.ptr.get().is_null() {
            self.ptr.set(self);
        } else {
            assert!(self.ptr.get() == self);
        }
    }
}

impl Drop for Foo {
    fn drop(&mut self) {
        self.touch();
    }
}

thread_local!(static FOO: Foo = Foo::default());

fn main() {
    thread::spawn(|| {
        FOO.with(|foo| foo.touch());
        FOO.with(|foo| foo.touch());
    })
    .join()
    .unwrap();
}
