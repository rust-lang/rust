// Regression test for an ICE that occurred with the universes code:
//
// The signature of the closure `|_|` was being inferred to
// `exists<'r> fn(&'r u8)`. This should result in a type error since
// the signature `for<'r> fn(&'r u8)` is required. However, due to a
// bug in the type variable generalization code, the placeholder for
// `'r` was leaking out into the writeback phase, causing an ICE.

trait ClonableFn<T> {
    fn clone(&self) -> Box<dyn Fn(T)>;
}

impl<T, F: 'static> ClonableFn<T> for F
where
    F: Fn(T) + Clone,
{
    fn clone(&self) -> Box<dyn Fn(T)> {
        Box::new(self.clone())
    }
}

struct Foo(Box<dyn for<'a> ClonableFn<&'a bool>>);

fn main() {
    Foo(Box::new(|_| ())); //~ ERROR implementation of `FnOnce` is not general enough
}
