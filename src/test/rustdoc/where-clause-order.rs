#![crate_name = "foo"]

pub trait SomeTrait<Rhs = Self>
where
    Rhs: ?Sized,
{
}

// @has 'foo/trait.SomeTrait.html'
// @has - "//*[@id='impl-SomeTrait%3C(A%2C%20B%2C%20C%2C%20D%2C%20E)%3E-for-(A%2C%20B%2C%20C%2C%20D%2C%20E)']/h3" "impl<A, B, C, D, E> SomeTrait<(A, B, C, D, E)> for (A, B, C, D, E)where A: PartialOrd<A> + PartialEq<A>, B: PartialOrd<B> + PartialEq<B>, C: PartialOrd<C> + PartialEq<C>, D: PartialOrd<D> + PartialEq<D>, E: PartialOrd<E> + PartialEq<E> + ?Sized, "
impl<A, B, C, D, E> SomeTrait<(A, B, C, D, E)> for (A, B, C, D, E)
where
    A: PartialOrd<A> + PartialEq<A>,
    B: PartialOrd<B> + PartialEq<B>,
    C: PartialOrd<C> + PartialEq<C>,
    D: PartialOrd<D> + PartialEq<D>,
    E: PartialOrd<E> + PartialEq<E> + ?Sized,
{
}
