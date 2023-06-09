#![crate_type = "lib"]

static mut COUNT: u64 = 1;

pub fn get_count() -> u64 { unsafe { COUNT } }

#[derive(Copy, Clone)]
pub struct Foo;

impl Foo {
    pub fn foo(self, x: &Foo) {
        unsafe { COUNT *= 2; }
        // Test internal call.
        Foo::bar(&self);
        Foo::bar(x);

        Foo::baz(self);
        Foo::baz(*x);

        Foo::qux(Box::new(self));
        Foo::qux(Box::new(*x));
    }

    pub fn bar(&self) {
        unsafe { COUNT *= 3; }
    }

    pub fn baz(self) {
        unsafe { COUNT *= 5; }
    }

    pub fn qux(self: Box<Foo>) {
        unsafe { COUNT *= 7; }
    }
}
