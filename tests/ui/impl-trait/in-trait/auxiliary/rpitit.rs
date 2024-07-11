use std::ops::Deref;

pub trait Foo {
    fn bar(self) -> impl Deref<Target = impl Sized>;
}

pub struct Foreign;
impl Foo for Foreign {
    #[expect(refining_impl_trait)]
    fn bar(self) -> &'static () {
        &()
    }
}
