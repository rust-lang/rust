//@ compile-flags: -Znext-solver
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

// Check that we correctly rerun the trait solver for heads of cycles,
// even if they are not the root.

struct A<T>(*const T);
struct B<T>(*const T);
struct C<T>(*const T);

#[rustc_coinductive]
trait Trait<'a, 'b> {}
trait NotImplemented {}

impl<'a, 'b, T> Trait<'a, 'b> for A<T> where B<T>: Trait<'a, 'b> {}

// With this the root of `B<T>` is `A<T>`, even if the other impl does
// not have a cycle with `A<T>`. This candidate never applies because of
// the `A<T>: NotImplemented` bound.
impl<'a, 'b, T> Trait<'a, 'b> for B<T>
where
    A<T>: Trait<'a, 'b>,
    A<T>: NotImplemented,
{
}

// HACK: This impls is necessary so that the impl above is well-formed.
//
// When checking that the impl above is well-formed we check `B<T>: Trait<'a, 'b>`
// with the where clauses `A<T>: Trait<'a, 'b>` and `A<T> NotImplemented`. Trying to
// use the impl itself to prove that adds region constraints as we uniquified the
// regions in the `A<T>: Trait<'a, 'b>` where-bound. As both the impl above
// and the impl below now apply with some constraints, we failed with ambiguity.
impl<'a, 'b, T> Trait<'a, 'b> for B<T>
where
    A<T>: NotImplemented,
{}

// This impl directly requires 'b to be equal to 'static.
//
// Because of the coinductive cycle through `C<T>` it also requires
// 'a to be 'static.
impl<'a, T> Trait<'a, 'static> for B<T>
where
    C<T>: Trait<'a, 'a>,
{}

// In the first iteration of `B<T>: Trait<'a, 'b>` we don't add any
// constraints here, only after setting the provisional result to require
// `'b == 'static` do we also add that constraint for `'a`.
impl<'a, 'b, T> Trait<'a, 'b> for C<T>
where
    B<T>: Trait<'a, 'b>,
{}

fn impls_trait<'a, 'b, T: Trait<'a, 'b>>() {}

fn check<'a, T>() {
    impls_trait::<'a, 'static, A<T>>();
    //~^ ERROR lifetime may not live long enough
}

fn main() {
    check::<()>();
}
