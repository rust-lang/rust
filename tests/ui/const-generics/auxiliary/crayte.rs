//@ edition:2018

pub trait Foo<const N: usize> {}
struct Local;
impl<const N: usize> Foo<N> for Local {}

pub fn out_foo<const N: usize>() -> impl Foo<N> { Local }
pub fn in_foo<const N: usize>(_: impl Foo<N>) {}

pub async fn async_simple<const N: usize>(_: [u8; N]) {}
pub async fn async_out_foo<const N: usize>() -> impl Foo<N> { Local }
pub async fn async_in_foo<const N: usize>(_: impl Foo<N>) {}

pub trait Bar<const N: usize> {
    type Assoc: Foo<N>;
}
