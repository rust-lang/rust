//@ compile-flags: -Znext-solver
#![feature(rustc_attrs)]

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
trait Trait<T: ?Sized, V: ?Sized, D: ?Sized> {}
struct A<T: ?Sized>(*const T);
struct B<T: ?Sized>(*const T);

trait IncompleteGuidance<T: ?Sized, V: ?Sized> {}
impl<T: ?Sized, U: ?Sized + 'static> IncompleteGuidance<U, u8> for T {}
impl<T: ?Sized, U: ?Sized + 'static> IncompleteGuidance<U, i8> for T {}
impl<T: ?Sized, U: ?Sized + 'static> IncompleteGuidance<U, i16> for T {}

trait ImplGuidance<T: ?Sized, V: ?Sized> {}
impl<T: ?Sized> ImplGuidance<u32, u8> for T {}
impl<T: ?Sized> ImplGuidance<i32, i8> for T {}

impl<T: ?Sized, U: ?Sized, V: ?Sized, D: ?Sized> Trait<U, V, D> for A<T>
where
    T: IncompleteGuidance<U, V>,
    A<T>: Trait<U, D, V>,
    B<T>: Trait<U, V, D>,
    (): ToU8<D>,
{
}

trait ToU8<T: ?Sized> {}
impl ToU8<u8> for () {}

impl<T: ?Sized, U: ?Sized, V: ?Sized, D: ?Sized> Trait<U, V, D> for B<T>
where
    T: ImplGuidance<U, V>,
    A<T>: Trait<U, V, D>,
{
}

fn impls_trait<T: ?Sized + Trait<U, V, D>, U: ?Sized, V: ?Sized, D: ?Sized>() {}

fn with_bound<X>()
where
    X: IncompleteGuidance<i32, u8>,
    X: IncompleteGuidance<u32, i8>,
    X: IncompleteGuidance<u32, i16>,
{
    impls_trait::<B<X>, _, _, _>(); // entering the cycle from `B` works

    // entering the cycle from `A` fails, but would work if we were to use the cache
    // result of `B<X>`.
    impls_trait::<A<X>, _, _, _>();
    //~^ ERROR the trait bound `A<X>: Trait<_, _, _>` is not satisfied
}

fn main() {
    with_bound::<u32>();
}
