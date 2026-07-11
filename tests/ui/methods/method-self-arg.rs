//@ run-pass
// Test method calls with self as an argument

use std::sync::atomic::{AtomicUsize, Ordering};

static COUNT: AtomicUsize = AtomicUsize::new(1);

fn multiply_count(factor: usize) {
    COUNT.try_update(Ordering::Relaxed, Ordering::Relaxed, |count| Some(count * factor)).unwrap();
}

#[derive(Copy, Clone)]
struct Foo;

impl Foo {
    fn foo(self, x: &Foo) {
        multiply_count(2);
        // Test internal call.
        Foo::bar(&self);
        Foo::bar(x);

        Foo::baz(self);
        Foo::baz(*x);

        Foo::qux(Box::new(self));
        Foo::qux(Box::new(*x));
    }

    fn bar(&self) {
        multiply_count(3);
    }

    fn baz(self) {
        multiply_count(5);
    }

    fn qux(self: Box<Foo>) {
        multiply_count(7);
    }
}

fn main() {
    let x = Foo;
    // Test external call.
    Foo::bar(&x);
    Foo::baz(x);
    Foo::qux(Box::new(x));

    x.foo(&x);

    assert_eq!(COUNT.load(Ordering::Relaxed), 2*3*3*3*5*5*5*7*7*7);
}
