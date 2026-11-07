pub struct Foo<'a, A:'a>(&'a A);

impl<'a, A> Foo<'a, A> {
    pub fn new(a: &'a A) -> Foo<'a, A> {
        Foo(a)
    }
}
