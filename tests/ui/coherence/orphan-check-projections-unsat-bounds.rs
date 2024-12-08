// This used to ICE in an earlier iteration of #117164.
// The normalization performed during orphan checking happens inside an empty ParamEnv and
// with type parameters mapped to fresh infer vars. Therefore it may fail for example due to
// unsatisfied bounds while normalization outside of orphan checking succeeds.

//@ revisions: classic next
//@[next] compile-flags: -Znext-solver

//@ check-pass
//@ aux-crate:foreign=parametrized-trait.rs
//@ edition:2021

struct Wrapper<T>(T);

trait Bound {}

trait Discard { type Output; }

impl<T> Discard for Wrapper<T>
where
    Wrapper<T>: Bound
{
    type Output = LocalTy;
}

struct LocalTy;

impl<T> foreign::Trait1<LocalTy, T> for <Wrapper<T> as Discard>::Output
//~^ WARNING type parameter `T` must be covered by another type
//~| WARNING this was previously accepted by the compiler
where
    Wrapper<T>: Bound
{}

fn main() {}
