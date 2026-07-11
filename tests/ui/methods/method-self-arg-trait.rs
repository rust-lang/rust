//@ run-pass
// Test method calls with self as an argument

use std::sync::atomic::{AtomicU64, Ordering};

static COUNT: AtomicU64 = AtomicU64::new(1);

fn multiply_count(factor: u64) {
    COUNT.try_update(Ordering::Relaxed, Ordering::Relaxed, |count| Some(count * factor)).unwrap();
}

#[derive(Copy, Clone)]
struct Foo;

trait Bar : Sized {
    fn foo1(&self);
    fn foo2(self);
    fn foo3(self: Box<Self>);

    fn bar1(&self) {
        multiply_count(7);
    }
    fn bar2(self) {
        multiply_count(11);
    }
    fn bar3(self: Box<Self>) {
        multiply_count(13);
    }
}

impl Bar for Foo {
    fn foo1(&self) {
        multiply_count(2);
    }

    fn foo2(self) {
        multiply_count(3);
    }

    fn foo3(self: Box<Foo>) {
        multiply_count(5);
    }
}

impl Foo {
    fn baz(self) {
        multiply_count(17);
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

    assert_eq!(COUNT.load(Ordering::Relaxed), 2*2*3*3*5*5*7*7*11*11*13*13*17);
}
