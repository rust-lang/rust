//@ run-pass
// Test method calls with self as an argument

use std::sync::atomic::{AtomicU64, Ordering};

static COUNT: AtomicU64 = AtomicU64::new(1);

#[derive(Copy, Clone)]
struct Foo;

trait Bar : Sized {
    fn foo1(&self);
    fn foo2(self);
    fn foo3(self: Box<Self>);

    fn bar1(&self) {
        COUNT.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |mut c| {
            c *= 7;
            Some(c)
        }).unwrap();
    }
    fn bar2(self) {
        COUNT.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |mut c| {
            c *= 11;
            Some(c)
        }).unwrap();
    }
    fn bar3(self: Box<Self>) {
        COUNT.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |mut c| {
            c *= 13;
            Some(c)
        }).unwrap();
    }
}

impl Bar for Foo {
    fn foo1(&self) {
        COUNT.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |mut c| {
            c *= 2;
            Some(c)
        }).unwrap();
    }

    fn foo2(self) {
        COUNT.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |mut c| {
            c *= 3;
            Some(c)
        }).unwrap();
    }

    fn foo3(self: Box<Foo>) {
        COUNT.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |mut c| {
            c *= 5;
            Some(c)
        }).unwrap();
    }
}

impl Foo {
    fn baz(self) {
        COUNT.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |mut c| {
            c *= 17;
            Some(c)
        }).unwrap();
        // Test internal call.
        Bar::foo1(&self);
        Bar::foo2(self);
        Bar::foo3(Box::new(self));

        Bar::bar1(&self);
        Bar::bar2(self);
        Bar::bar3(Box::new(self));
    }
}

fn main() {
    let x = Foo;
    // Test external call.
    Bar::foo1(&x);
    Bar::foo2(x);
    Bar::foo3(Box::new(x));

    Bar::bar1(&x);
    Bar::bar2(x);
    Bar::bar3(Box::new(x));

    x.baz();

    assert_eq!(COUNT.load(Ordering::SeqCst), 2*2*3*3*5*5*7*7*11*11*13*13*17);
}
