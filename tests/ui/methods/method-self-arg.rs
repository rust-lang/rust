//@ run-pass
// Test method calls with self as an argument

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

static mut COUNT: usize = 1;

#[derive(Copy, Clone)]
struct Foo;

impl Foo {
    fn foo(self, x: &Foo) {
        unsafe { COUNT *= 2; }
        // Test internal call.
        Foo::bar(&self);
        Foo::bar(x);

        Foo::baz(self);
        Foo::baz(*x);

        Foo::qux(Box::new(self));
        Foo::qux(Box::new(*x));
    }

    fn bar(&self) {
        unsafe { COUNT *= 3; }
    }

    fn baz(self) {
        unsafe { COUNT *= 5; }
    }

    fn qux(self: Box<Foo>) {
        unsafe { COUNT *= 7; }
    }
}

fn main() {
    let x = Foo;
    // Test external call.
    Foo::bar(&x);
    Foo::baz(x);
    Foo::qux(Box::new(x));

    x.foo(&x);

    unsafe { assert_eq!(COUNT, 2*3*3*3*5*5*5*7*7*7); }
}
