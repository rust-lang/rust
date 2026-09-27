// Regression test for issue #163320: probing associated-type suggestions
// must retain const-parameter typing information under the next solver.

//@ compile-flags: -Znext-solver=globally
//@ check-fail

#![feature(generic_const_parameter_types)]

fn where_clause<const N: usize>()
where
    <[(); N] as IntoIterator>::Item::Item:,
    //~^ ERROR ambiguous associated type
{}

fn nested_where<const N: usize>()
where
    Option<<[(); N] as IntoIterator>::Item::Item>: Sized,
    //~^ ERROR ambiguous associated type
{}

fn parameter<const N: usize>(
    _: <[(); N] as IntoIterator>::Item::Item,
    //~^ ERROR ambiguous associated type
) {}

type Alias<const N: usize> =
    <[(); N] as IntoIterator>::Item::Item;
//~^ ERROR ambiguous associated type

struct Field<const N: usize> {
    field: <[(); N] as IntoIterator>::Item::Item,
    //~^ ERROR ambiguous associated type
}

struct Parent<const N: usize>;

impl<const N: usize> Parent<N> {
    fn inherited()
    where
        <[(); N] as IntoIterator>::Item::Item:,
        //~^ ERROR ambiguous associated type
    {}
}

struct ConstParameter<
    const N: usize,
    const M: <[(); N] as IntoIterator>::Item::Item,
    //~^ ERROR ambiguous associated type
>;

struct TypeParameterDefault<
    const N: usize,
    T = <[(); N] as IntoIterator>::Item::Item,
    //~^ ERROR ambiguous associated type
>(std::marker::PhantomData<T>);

fn main() {}
