//@ check-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver

// Regression test for nalgebra hang from
//     https://github.com/rust-lang/rust/pull/130654#issuecomment-2365465354
trait HasAlias {}

struct Dummy;
trait DummyTrait {
    type DummyType<T: HasAlias>;
}
impl DummyTrait for Dummy {
    type DummyType<T: HasAlias> = T;
}
type AliasOf<T> = <Dummy as DummyTrait>::DummyType<T>;

struct Matrix<T, S>(T, S);
type OMatrix<T> = Matrix<T, AliasOf<T>>;

impl<T: HasAlias> HasAlias for OMatrix<T> {}

trait SimdValue {
    type Element;
}
impl<T: HasAlias + SimdValue<Element: HasAlias>> SimdValue for OMatrix<T> {
    type Element = OMatrix<T::Element>;
}

trait Unimplemented {}
pub trait MyFrom<T> {}
impl<T: Unimplemented> MyFrom<T> for T {}
impl<T: SimdValue<Element: HasAlias>> MyFrom<T> for OMatrix<T::Element> {}

fn main() {}
