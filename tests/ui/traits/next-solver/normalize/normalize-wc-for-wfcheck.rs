// Wfcheck normalizes where-clauses when checking for WF, so this test passes with the old solver,
// even though it probably shouldn't due to `repro` missing a `T: 'a` bound. For now, until wfcheck
// checks clauses pre-normalization, the new solver matches the old solver.

// This test is extracted from a new solver crater run error encountered in `modcholesky`.

//@ check-pass
//@ compile-flags: -Znext-solver

pub struct View<A>(A);
pub trait Data {
    type Elem;
}
impl<'a, A> Data for View<&'a A> {
    type Elem = A;
}

pub fn repro<'a, T>()
where
    <View<&'a T> as Data>::Elem: Sized,
{
}

fn main() {}
