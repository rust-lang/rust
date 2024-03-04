// Regression test for issue #121898.

trait Base {
    type Base<B>;
}

trait Functor<A>: Base {
    fn fmap<B>(self, f: impl Fn(A) -> B) -> Self::Base<B>
    where
        Self::Base<B>: Functor<B>;
}

fn fmap2<T, A, B, C>(input: T, f1: impl Fn(A) -> B, f2: impl Fn(B) -> C) -> T::Base<C>
where
    T: Functor<A>,
    T::Base<B>: Functor<B, Base<C> = T::Base<C>>,
{
    input.fmap(f1).fmap(f2)
    //~^ ERROR the trait bound `<T as Base>::Base<C>: Functor<C>` is not satisfied
}

fn main() {}
