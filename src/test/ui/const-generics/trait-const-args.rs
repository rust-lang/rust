// check-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct Const<const N: usize>;
trait Foo<const N: usize> {}

impl<const N: usize> Foo<N> for Const<N> {}

fn foo_impl(_: impl Foo<3>) {}

fn foo_explicit<T: Foo<3>>(_: T) {}

fn foo_where<T>(_: T)
where
    T: Foo<3>,
{
}

fn main() {
    foo_impl(Const);
    foo_impl(Const::<3>);

    foo_explicit(Const);
    foo_explicit(Const::<3>);

    foo_where(Const);
    foo_where(Const::<3>);
}
