// Test that when we have a `<T as MyTrait<'a>>::Output: 'a`
// relationship in the environment we take advantage of it.  In this
// case, that means we **don't** have to prove that `T: 'a`.
//
// Regression test for #53121.
//
// build-pass (FIXME(62277): could be check-pass?)

trait MyTrait<'a> {
    type Output;
}

fn foo<'a, T>() -> &'a ()
where
    T: MyTrait<'a>,
    <T as MyTrait<'a>>::Output: 'a,
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
