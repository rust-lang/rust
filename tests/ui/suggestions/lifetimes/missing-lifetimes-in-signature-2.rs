// Regression test for #81650
//@ run-rustfix

#![allow(warnings)]

struct Foo<'a> {
    x: &'a mut &'a i32,
}

impl<'a> Foo<'a> {
    fn bar<F, T>(&self, f: F)
    where
        F: FnOnce(&Foo<'a>) -> T,
        F: 'a,
    {}
}

trait Test {
    fn test(&self);
}

fn func<T: Test>(_dummy: &Foo, foo: &Foo, t: T) {
    foo.bar(move |_| {
    //~^ ERROR the parameter type `T` may not live long enough
        t.test();
    });
}

// Test that the suggested fix does not overconstrain `func`. See #115375.
fn test_func<'a, T: Test + 'a>(dummy: &Foo, foo: &Foo<'a>, t: T) {
    func(dummy, foo, t);
}

fn main() {}
