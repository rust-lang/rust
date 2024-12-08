#![crate_name = "foo"]

pub trait SomeTrait<Rhs = Self>
where
    Rhs: ?Sized,
{
}

//@ has 'foo/trait.SomeTrait.html'
//@ has - "//*[@id='impl-SomeTrait-for-(A,+B,+C,+D,+E)']/h3" "impl<A, B, C, D, E> SomeTrait for (A, B, C, D, E)where A: PartialOrd<A> + PartialEq<A>, B: PartialOrd<B> + PartialEq<B>, C: PartialOrd<C> + PartialEq<C>, D: PartialOrd<D> + PartialEq<D>, E: PartialOrd<E> + PartialEq<E> + ?Sized, "
impl<A, B, C, D, E> SomeTrait<(A, B, C, D, E)> for (A, B, C, D, E)
where
    A: PartialOrd<A> + PartialEq<A>,
    B: PartialOrd<B> + PartialEq<B>,
    C: PartialOrd<C> + PartialEq<C>,
    D: PartialOrd<D> + PartialEq<D>,
    E: PartialOrd<E> + PartialEq<E> + ?Sized,
{
}

//@ has - "//*[@id='impl-SomeTrait%3C(A,+B,+C,+D)%3E-for-(A,+B,+C,+D,+E)']/h3" "impl<A, B, C, D, E> SomeTrait<(A, B, C, D)> for (A, B, C, D, E)where A: PartialOrd<A> + PartialEq<A>, B: PartialOrd<B> + PartialEq<B>, C: PartialOrd<C> + PartialEq<C>, D: PartialOrd<D> + PartialEq<D>, E: PartialOrd<E> + PartialEq<E> + ?Sized, "
impl<A, B, C, D, E> SomeTrait<(A, B, C, D)> for (A, B, C, D, E)
where
    A: PartialOrd<A> + PartialEq<A>,
    B: PartialOrd<B> + PartialEq<B>,
    C: PartialOrd<C> + PartialEq<C>,
    D: PartialOrd<D> + PartialEq<D>,
    E: PartialOrd<E> + PartialEq<E> + ?Sized,
{
}
