//@ known-bug: #137865
trait Foo {
    type Assoc<const N: Self>;
    fn foo() -> Self::Assoc<3>;
}
