pub struct Foo<T> {
    x: T,
}

pub trait Bar {
    type Fuu;

    fn foo(foo: Self::Fuu);
}

// @has doc_assoc_item/struct.Foo.html '//*[@class="impl"]' 'impl<T: Bar<Fuu = u32>> Foo<T>'
impl<T: Bar<Fuu = u32>> Foo<T> {
    pub fn new(t: T) -> Foo<T> {
        Foo {
            x: t,
        }
    }
}
