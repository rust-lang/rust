// Test that we are NOT able to establish that `<T as
// MyTrait<'a>>::Output: 'a` outlives `'a` here -- we have only one
// recourse, which is to prove that `T: 'a` and `'a: 'a`, but we don't
// know that `T: 'a`.

trait MyTrait<'a> {
    type Output;
}

fn foo<'a, T>() -> &'a ()
where
    T: MyTrait<'a>,
{
    bar::<T::Output>() //~ ERROR the parameter type `T` may not live long enough
}

fn bar<'a, T>() -> &'a ()
where
    T: 'a,
{
    &()
}

fn main() {}
