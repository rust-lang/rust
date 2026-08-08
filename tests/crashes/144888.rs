//@ known-bug: rust-lang/rust#144888
trait Super {
    type Assoc;
}
impl dyn Foo<()> {}
trait Foo<T>: Super<Assoc = T>
where
    <Self as Mirror>::Assoc: Clone,
{
    fn transmute(&self) {}
}

trait Mirror {
    type Assoc;
}
impl<T: Super<Assoc = ()>> Mirror for T {}
