// Cited from [#157937](https://github.com/rust-lang/rust/issues/157937)

#![feature(generic_const_exprs)]
trait Trait {
    const CONST: usize;
}
struct A<T> {
    _marker: T,
}
impl<const N: usize> Trait for [i8; N] {
    const CONST: usize = N;
}
impl<const N: usize> From<usize> for A<[i8; N]> {
    fn from(_: usize) -> Self {
        todo!()
    }
}
impl<T: Trait> From<A<[i8; T::CONST]>> for A<T> {
//~^ ERROR: conflicting implementations of trait `From<A<[i8; _]>>` for type `A<[i8; _]>` [E0119]
    fn from(_: A<[i8; T::CONST]>) -> Self {
        todo!()
    }
}
fn f<T: Trait>() -> A<T>
where
    [(); T::CONST]:,
{
    let a = A::from(0);
    A::from(a)
}
fn main() {
    f::<[i8; 1]>();
}
