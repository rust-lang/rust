//@ run-pass
//@ compile-flags: -C lto
//@ no-prefer-dynamic
//@ needs-threads

use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};

static HIT: AtomicUsize = AtomicUsize::new(0);

thread_local!(static A: Foo = Foo);

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        HIT.fetch_add(1, Ordering::Relaxed);
    }
}

fn main() {
    assert_eq!(HIT.load(Ordering::Relaxed), 0);
    thread::spawn(|| {
        assert_eq!(HIT.load(Ordering::Relaxed), 0);
        A.with(|_| ());
        assert_eq!(HIT.load(Ordering::Relaxed), 0);
    }).join().unwrap();
    assert_eq!(HIT.load(Ordering::Relaxed), 1);
}
