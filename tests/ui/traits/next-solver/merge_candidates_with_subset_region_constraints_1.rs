//@ edition: 2024
//@ compile-flags: -Znext-solver

// We merge param env and alias bound candidates if their only differences are region constraints
// and one's region constraints is a superset of another's.
// This checks that we don't do that for impl candidates.

#![feature(min_specialization)]

trait Bound {
    fn dummy() {}
}
fn impls_bound<T: Bound>() {}

impl<'a, T> Bound for (T, &'a i32) where T: 'a {
    default fn dummy() {}
}

impl<'a> Bound for ((), &'a i32) {
    fn dummy() {}
}

fn do_not_merge_overlapping_impl_candidates()
{
    impls_bound::<((), _)>();
    //~^ ERROR: type annotations needed [E0283]
}

fn main() {}
