//@ check-pass
//@ compile-flags: --crate-type=lib
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Since #120752, also get alias-bound candidates from a nested self-type, so prefering
// alias-bound over where-bound candidates can be incorrect. This test checks that case and that
// we prefer non-nested alias-bound candidates over where-bound candidates over nested alias-bound
// candidates.

trait OtherTrait<'a> {
    type Assoc: ?Sized;
}

trait Trait
where
    <Self::Assoc as OtherTrait<'static>>::Assoc: Sized,
{
    type Assoc: for<'a> OtherTrait<'a>;
}

fn impls_sized<T: Sized>() {}
fn foo<'a, T>()
where
    T: Trait,
    for<'hr> <T::Assoc as OtherTrait<'hr>>::Assoc: Sized,
{
    impls_sized::<<T::Assoc as OtherTrait<'a>>::Assoc>();
}
