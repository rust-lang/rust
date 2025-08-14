trait Foo<A> {
    fn get(&self, A: &A) {}
}

trait Bar {
    type Out;
}

impl<T> Foo<T> for [isize; 0] {
    // OK, T is used in `Foo<T>`.
}

impl<T, U> Foo<T> for [isize; 1] {
    //~^ ERROR the type parameter `U` is not constrained
}

impl<T, U> Foo<T> for [isize; 2]
where
    T: Bar<Out = U>,
{
    // OK, `U` is now constrained by the output type parameter.
}

impl<T: Bar<Out = U>, U> Foo<T> for [isize; 3] {
    // OK, same as above but written differently.
}

impl<T, U> Foo<T> for U {
    //~^ ERROR conflicting implementations of trait `Foo<_>` for type `[isize; 0]`
}

impl<T, U> Bar for T {
    //~^ ERROR the type parameter `U` is not constrained

    type Out = U;

    // Using `U` in an associated type within the impl is not good enough!
}

impl<T, U> Bar for T
where
    T: Bar<Out = U>,
{
    //~^^^^ ERROR the type parameter `U` is not constrained by the impl trait, self type, or predicates
    //~| ERROR conflicting implementations of trait `Bar`
    // This crafty self-referential attempt is still no good.
}

impl<T, U, V> Foo<T> for T
where
    (T, U): Bar<Out = V>,
{
    //~^^^^ ERROR the type parameter `U` is not constrained
    //~| ERROR the type parameter `V` is not constrained
    //~| ERROR conflicting implementations of trait `Foo<[isize; 0]>` for type `[isize; 0]`

    // Here, `V` is bound by an output type parameter, but the inputs
    // are not themselves constrained.
}

impl<T, U, V> Foo<(T, U)> for T
where
    (T, U): Bar<Out = V>,
{
    //~^^^^ ERROR conflicting implementations of trait `Foo<([isize; 0], _)>` for type `[isize; 0]`
    // As above, but both T and U ARE constrained.
}

fn main() {}
