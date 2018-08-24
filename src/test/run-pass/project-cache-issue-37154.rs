// Regression test for #37154: the problem here was that the cache
// results in a false error because it was caching skolemized results
// even after those skolemized regions had been popped.

trait Foo {
    fn method(&self) {}
}

struct Wrapper<T>(T);

impl<T> Foo for Wrapper<T> where for<'a> &'a T: IntoIterator<Item=&'a ()> {}

fn f(x: Wrapper<Vec<()>>) {
    x.method(); // This works.
    x.method(); // error: no method named `method`
}

fn main() { }
