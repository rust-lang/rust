// Regression test for #81650

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

fn func<T: Test>(foo: &Foo, t: T) {
    foo.bar(move |_| {
    //~^ ERROR the parameter type `T` may not live long enough
        t.test();
    });
}

fn main() {}
