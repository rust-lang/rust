//@ run-pass
// Test method calls with self as an argument

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

static mut COUNT: u64 = 1;

#[derive(Copy, Clone)]
struct Foo;

trait Bar : Sized {
    fn foo1(&self);
    fn foo2(self);
    fn foo3(self: Box<Self>);

    fn bar1(&self) {
        unsafe { COUNT *= 7; }
    }
    fn bar2(self) {
        unsafe { COUNT *= 11; }
    }
    fn bar3(self: Box<Self>) {
        unsafe { COUNT *= 13; }
    }
}

impl Bar for Foo {
    fn foo1(&self) {
        unsafe { COUNT *= 2; }
    }

    fn foo2(self) {
        unsafe { COUNT *= 3; }
    }

    fn foo3(self: Box<Foo>) {
        unsafe { COUNT *= 5; }
    }
}

impl Foo {
    fn baz(self) {
        unsafe { COUNT *= 17; }
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

    unsafe { assert_eq!(COUNT, 2*2*3*3*5*5*7*7*11*11*13*13*17); }
}
