//@ run-pass

use std::sync::atomic::{AtomicBool, Ordering};

static DROP_RAN: AtomicBool = AtomicBool::new(false);

trait Bar {
    fn do_something(&mut self); //~ WARN method `do_something` is never used
}

struct BarImpl;

impl Bar for BarImpl {
    fn do_something(&mut self) {}
}


struct Foo<B: Bar>(#[allow(dead_code)] B);

impl<B: Bar> Drop for Foo<B> {
    fn drop(&mut self) {
        DROP_RAN.store(true, Ordering::Relaxed);
    }
}


fn main() {
    {
       let _x: Foo<BarImpl> = Foo(BarImpl);
    }
    assert_eq!(DROP_RAN.load(Ordering::Relaxed), true);
}
