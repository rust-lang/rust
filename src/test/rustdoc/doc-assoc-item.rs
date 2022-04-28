pub struct Foo<T> {
    x: T,
}

pub trait Bar {
    type Fuu;

    fn foo(foo: Self::Fuu);
}

// @has doc_assoc_item/struct.Foo.html
// @has - '//*[@class="impl has-srclink"]' 'impl<T> Foo<T>'
// @has - '//*[@class="impl has-srclink"]' 'where T: Bar<Fuu = u32>'
impl<T: Bar<Fuu = u32>> Foo<T> {
    pub fn new(t: T) -> Foo<T> {
        Foo { x: t }
    }
}
