pub struct Foo<T>(T);

impl<T> Foo<T> {
    pub fn new(t: T) -> Foo<T> {
        Foo(t)
    }
}
