// run-pass
#![allow(stable_features)]

// ignore-emscripten no threads support

#![feature(thread_local_try_with)]

use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Foo { cnt: usize }

thread_local!(static FOO: Foo = Foo::init());

static CNT: AtomicUsize = AtomicUsize::new(0);

impl Foo {
    fn init() -> Foo {
        let cnt = CNT.fetch_add(1, Ordering::SeqCst);
        if cnt == 0 {
            FOO.with(|_| {});
        }
        Foo { cnt: cnt }
    }
}

impl Drop for Foo {
    fn drop(&mut self) {
        if self.cnt == 1 {
            FOO.with(|foo| assert_eq!(foo.cnt, 0));
        } else {
            assert_eq!(self.cnt, 0);
            if FOO.try_with(|_| ()).is_ok() {
                panic!("should not be in valid state");
            }
        }
    }
}

fn main() {
    thread::spawn(|| {
        FOO.with(|_| {});
    }).join().unwrap();
}
