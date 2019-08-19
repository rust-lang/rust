// run-pass
#![allow(stable_features)]

// ignore-emscripten no threads support

#![feature(thread_local_try_with)]

use std::thread;

static mut DROP_RUN: bool = false;

struct Foo;

thread_local!(static FOO: Foo = Foo {});

impl Drop for Foo {
    fn drop(&mut self) {
        assert!(FOO.try_with(|_| panic!("`try_with` closure run")).is_err());
        unsafe { DROP_RUN = true; }
    }
}

fn main() {
    thread::spawn(|| {
        assert_eq!(FOO.try_with(|_| {
            132
        }).expect("`try_with` failed"), 132);
    }).join().unwrap();
    assert!(unsafe { DROP_RUN });
}
