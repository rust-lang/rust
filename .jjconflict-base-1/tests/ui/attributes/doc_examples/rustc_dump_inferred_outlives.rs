//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

trait Trait<'x, T>
where
    T: 'x,
{
    type Type;
}

#[rustc_dump_inferred_outlives]
struct Foo<'a, A, B>
where
    A: Trait<'a, B>,
{
    foo: <A as Trait<'a, B>>::Type,
}
