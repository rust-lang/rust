trait Trait<T> {
    type Type<'a> where T: 'a;
    fn foo(x: &T) -> Self::Type<'_>;
}
impl<T> Trait<T> for () {
    type Type<'a> where T: 'a = &'a T;
    fn foo(x: &T) -> Self::Type<'_> {
        x
    }
}
