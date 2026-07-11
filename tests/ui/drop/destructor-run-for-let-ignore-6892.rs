// https://github.com/rust-lang/rust/issues/6892
//@ run-pass
#![allow(dead_code)]
// Ensures that destructors are run for expressions of the form "let _ = e;"
// where `e` is a type which requires a destructor.

use std::sync::atomic::{AtomicUsize, Ordering};

struct Foo;
struct Bar { x: isize }
struct Baz(isize);
enum FooBar { _Foo(Foo), _Bar(usize) }

static NUM_DROPS: AtomicUsize = AtomicUsize::new(0);

impl Drop for Foo {
    fn drop(&mut self) {
        NUM_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}
impl Drop for Bar {
    fn drop(&mut self) {
        NUM_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}
impl Drop for Baz {
    fn drop(&mut self) {
        NUM_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}
impl Drop for FooBar {
    fn drop(&mut self) {
        NUM_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

fn main() {
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 0);
    { let _x = Foo; }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 1);
    { let _x = Bar { x: 21 }; }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 2);
    { let _x = Baz(21); }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 3);
    { let _x = FooBar::_Foo(Foo); }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 5);
    { let _x = FooBar::_Bar(42); }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 6);

    { let _ = Foo; }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 7);
    { let _ = Bar { x: 21 }; }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 8);
    { let _ = Baz(21); }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 9);
    { let _ = FooBar::_Foo(Foo); }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 11);
    { let _ = FooBar::_Bar(42); }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 12);
}
