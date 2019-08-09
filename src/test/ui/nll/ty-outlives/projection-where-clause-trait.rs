#![feature(nll)]

// Test that we are able to establish that `<T as
// MyTrait<'a>>::Output: 'a` outlives `'a` (because the trait says
// so).
//
// build-pass (FIXME(62277): could be check-pass?)

trait MyTrait<'a> {
    type Output: 'a;
}

fn foo<'a, T>() -> &'a ()
where
    T: MyTrait<'a>,
{
    bar::<T::Output>()
}

fn bar<'a, T>() -> &'a ()
where
    T: 'a,
{
    &()
}

fn main() {}
