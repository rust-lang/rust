trait Foo<A> {
    fn get(&self, A: &A) { }
}

trait Bar {
    type Out;
}

impl<T> Foo<T> for [isize;0] {
    // OK, T is used in `Foo<T>`.
}

impl<T,U> Foo<T> for [isize;1] {
    //~^ ERROR the type parameter `U` is not constrained
}

impl<T,U> Foo<T> for [isize;2] where T : Bar<Out=U> {
    // OK, `U` is now constrained by the output type parameter.
}

impl<T:Bar<Out=U>,U> Foo<T> for [isize;3] {
    // OK, same as above but written differently.
}

impl<T,U> Foo<T> for U {
    // OK, T, U are used everywhere. Note that the coherence check
    // hasn't executed yet, so no errors about overlap.
}

impl<T,U> Bar for T {
    //~^ ERROR the type parameter `U` is not constrained

    type Out = U;

    // Using `U` in an associated type within the impl is not good enough!
}

impl<T,U> Bar for T
    where T : Bar<Out=U>
{
    //~^^^ ERROR the type parameter `U` is not constrained

    // This crafty self-referential attempt is still no good.
}

impl<T,U,V> Foo<T> for T
    where (T,U): Bar<Out=V>
{
    //~^^^ ERROR the type parameter `U` is not constrained
    //~|   ERROR the type parameter `V` is not constrained

    // Here, `V` is bound by an output type parameter, but the inputs
    // are not themselves constrained.
}

impl<T,U,V> Foo<(T,U)> for T
    where (T,U): Bar<Out=V>
{
    // As above, but both T and U ARE constrained.
}

fn main() { }
