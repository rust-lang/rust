//! Regression test for <https://github.com/rust-lang/rust/issues/151503>.
//!
//! The overlap check for the two `MyFrom` impls used to hang.

trait HasAlias {}
struct Dummy;
trait DummyTrait {
    type DummyType<T>;
}
impl DummyTrait for Dummy {
    type DummyType<T> = T;
}
type AliasOf<T> = <Dummy as DummyTrait>::DummyType<T>;
struct Matrix<T, S>(T, S);
type OMatrix<T> = Matrix<T, AliasOf<T>>;

trait SimdValue {
    type Element;
}
impl<T: HasAlias + SimdValue> SimdValue for OMatrix<T> {
    type Element = OMatrix<T::Element>;
}
trait MyFrom<T> {}
impl<T> MyFrom<T> for T {}
impl<T: SimdValue> MyFrom<T> for OMatrix<T::Element> {}
//~^ ERROR conflicting implementations of trait `MyFrom<Matrix<_, _>>` for type `Matrix<_, _>`
impl<T, S> HasAlias for Matrix<T, S> {}

fn main() {}
