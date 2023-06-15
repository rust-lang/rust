// build-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::marker::PhantomData;

fn main() {
    let x = FooImpl::<BarImpl<1>> { phantom: PhantomData };
    x.foo::<BarImpl<1>>();
}

trait Foo<T>
where
    T: Bar,
{
    fn foo<U>(&self)
    where
        T: Operation<U>,
        <T as Operation<U>>::Output: Bar;
}

struct FooImpl<T>
where
    T: Bar,
{
    phantom: PhantomData<T>,
}

impl<T> Foo<T> for FooImpl<T>
where
    T: Bar,
{
    fn foo<U>(&self)
    where
        T: Operation<U>,
        <T as Operation<U>>::Output: Bar,
    {
        <<T as Operation<U>>::Output as Bar>::error_occurs_here();
    }
}

trait Bar {
    fn error_occurs_here();
}

struct BarImpl<const N: usize>;

impl<const N: usize> Bar for BarImpl<N> {
    fn error_occurs_here() {}
}

trait Operation<Rhs> {
    type Output;
}

//// Part-A: This causes error.
impl<const M: usize, const N: usize> Operation<BarImpl<M>> for BarImpl<N>
where
    BarImpl<{ N + M }>: Sized,
{
    type Output = BarImpl<{ N + M }>;
}

//// Part-B: This doesn't cause error.
// impl<const M: usize, const N: usize> Operation<BarImpl<M>> for BarImpl<N> {
//     type Output = BarImpl<M>;
// }

//// Part-C: This also doesn't cause error.
// impl<const M: usize, const N: usize> Operation<BarImpl<M>> for BarImpl<N> {
//     type Output = BarImpl<{ M }>;
// }
