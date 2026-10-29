//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// A minimization of an ambiguity error in `icu_provider`.
//
// cc trait-system-refactor-initiative#110

trait Yokeable<'a> {
    type Output;
}
trait Id {
    type Refl;
}

fn into_deserialized<M: Id>() -> M
where
    M::Refl: for<'a> Yokeable<'a>,
{
    try_map_project::<M, _>(|_| todo!())
}

fn try_map_project<M: Id, F>(_f: F) -> M
where
    M::Refl: for<'a> Yokeable<'a>,
    F: for<'a> FnOnce(&'a ()) -> <<M as Id>::Refl as Yokeable<'a>>::Output,
{
    todo!()
}

fn main() {}
