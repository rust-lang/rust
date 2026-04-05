//@ edition: 2024
//@ compile-flags: -Znext-solver
//@ check-pass

// A regression test for trait-system-refactor-initiative#265.
// We can have duplicate param env candidates modulo region constraints due to
// lazy param env normalization. This previouly prevents us from merging them and
// causes ambiguity.
// Now we merge param env and alias bound candidates if one's region constraints
// is a superset of another's.

// The region parameter is needed to disable `always-applicable` candidates.
trait Bound<'a> {}
fn impls_bound<'a, T: Bound<'a>>() {}

trait Id {
    type SelfType<'a>
    where
        Self: 'a;
}

fn test_single_generic_region_param<'a, T>()
where
    T: 'a,
    T: Id<SelfType<'a> = T>,
    T::SelfType<'a>: Bound<'a>,
    T: Bound<'a>,

{
    impls_bound::<'_, T>();
}

fn test_single_higher_ranked_region<'a, T>()
where
    T: 'static,
    for<'b> T: Id<SelfType<'b> = T>,
    for<'b> T::SelfType<'b>: Bound<'a>,
    T: Bound<'a>,
{
    impls_bound::<'_, T>();
}

trait Id1 {
    type SelfType<'a, 'b>
    where
        Self: 'a,
        Self: 'b;
}

trait Id2 {
    type SelfType<'a, 'b, 'c>
    where
        Self: 'a,
        Self: 'b,
        Self: 'c;
}

fn test_multiple_higher_ranked_region<'a, T>()
where
    T: 'static,
    T: Bound<'a>,
    for<'b> T: Id<SelfType<'b> = T>,
    for<'b> <T as Id>::SelfType<'b>: Bound<'a>,
    for<'b, 'c> T: Id1<SelfType<'b, 'c> = T>,
    for<'b, 'c> <T as Id1>::SelfType<'b, 'c>: Bound<'a>,
    for<'b, 'c, 'd> T: Id2<SelfType<'b, 'c, 'd> = T>,
    for<'b, 'c, 'd> <T as Id2>::SelfType<'b, 'c, 'd>: Bound<'a>,
{
    impls_bound::<'_, T>();
}

trait Id3 {
    type SelfType<'a, 'b>
    where
        Self: 'a,
        'a: 'b;
}

trait Id4 {
    type SelfType<'a, 'b, 'c>
    where
        Self: 'a,
        'a: 'b,
        'b: 'c;
}

fn test_multiple_higher_ranked_region_with_region_outlives<'a, T>()
where
    T: 'static,
    T: Bound<'a>,
    for<'b> T: Id<SelfType<'b> = T>,
    for<'b> <T as Id>::SelfType<'b>: Bound<'a>,
    for<'b, 'c> T: Id3<SelfType<'b, 'c> = T>,
    for<'b, 'c> <T as Id3>::SelfType<'b, 'c>: Bound<'a>,
    for<'b, 'c, 'd> T: Id4<SelfType<'b, 'c, 'd> = T>,
    for<'b, 'c, 'd> <T as Id4>::SelfType<'b, 'c, 'd>: Bound<'a>,
{
    impls_bound::<'_, T>();
}

fn main() {}
