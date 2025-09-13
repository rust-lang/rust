pub trait Trait {
    fn foo(_: impl Sized);
    fn bar<T>(_: impl Sized);
    fn baz<'a, const N: usize>();
    fn quux<'a: 'b, 'b, T: ?Sized>();
}
