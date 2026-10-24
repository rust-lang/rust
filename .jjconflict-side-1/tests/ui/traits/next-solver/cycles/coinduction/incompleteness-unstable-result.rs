//@ revisions: with without
//@ compile-flags: -Znext-solver
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

// This test is incredibly subtle. At its core the goal is to get a coinductive cycle,
// which, depending on its root goal, either holds or errors. We achieve this by getting
// incomplete inference via a `ParamEnv` candidate in the `A<T>` impl and required
// inference from an `Impl` candidate in the `B<T>` impl.
//
// To make global cache accesses stronger than the guidance from the where-bounds, we add
// another coinductive cycle from `A<T>: Trait<U, V, D>` to `A<T>: Trait<U, D, V>` and only
// constrain `D` directly. This means that any candidates which rely on `V` only make
// progress in the second iteration, allowing a cache access in the first iteration to take
// precedence.
//
// tl;dr: our caching of coinductive cycles was broken and this is a regression
// test for that.

#[rustc_coinductive]
trait Trait<T, V, D> {}
struct A<T>(*const T);
struct B<T>(*const T);

trait IncompleteGuidance<T, V> {}
impl<T, U: 'static> IncompleteGuidance<U, u8> for T {}
impl<T, U: 'static> IncompleteGuidance<U, i8> for T {}
impl<T, U: 'static> IncompleteGuidance<U, i16> for T {}

trait ImplGuidance<T, V> {}
impl<T> ImplGuidance<u32, u8> for T {}
impl<T> ImplGuidance<i32, i8> for T {}

impl<T, U, V, D> Trait<U, V, D> for A<T>
where
    T: IncompleteGuidance<U, V>,
    A<T>: Trait<U, D, V>,
    B<T>: Trait<U, V, D>,
    (): ToU8<D>,
{
}

trait ToU8<T> {}
impl ToU8<u8> for () {}

impl<T, U, V, D> Trait<U, V, D> for B<T>
where
    T: ImplGuidance<U, V>,
    A<T>: Trait<U, V, D>,
{
}

fn impls_trait<T: Trait<U, V, D>, U, V, D>() {}

fn with_bound<X>()
where
    X: IncompleteGuidance<i32, u8>,
    X: IncompleteGuidance<u32, i8>,
    X: IncompleteGuidance<u32, i16>,
{
    #[cfg(with)]
    impls_trait::<B<X>, _, _, _>(); // entering the cycle from `B` works

    // entering the cycle from `A` fails, but would work if we were to use the cache
    // result of `B<X>`.
    impls_trait::<A<X>, _, _, _>();
    //~^ ERROR the trait bound `A<X>: Trait<_, _, _>` is not satisfied
}

fn main() {
    with_bound::<u32>();
}
