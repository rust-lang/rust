pub trait Trait {
    fn foo(_: impl Sized);
    fn bar<T>(_: impl Sized);
}
