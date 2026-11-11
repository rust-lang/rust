// Test that if we need to prove that `<T as MyTrait<'a>>::Output:
// 'a`, but we only know that `<T as MyTrait<'b>>::Output: 'a`, that
// doesn't suffice.

trait MyTrait<'a> {
    type Output;
}

fn foo1<'a, 'b, T>() -> &'a ()
where
    for<'x> T: MyTrait<'x>,
    <T as MyTrait<'b>>::Output: 'a,
{
    bar::<<T as MyTrait<'a>>::Output>()
    //~^ ERROR the associated type `<T as MyTrait<'a>>::Output` may not live long enough
}

fn bar<'a, T>() -> &'a ()
where
    T: 'a,
{
    &()
}

fn main() {}
