//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// We've got `'0 member ['a, 'b, 'static]` and `'1 member ['a, 'b, 'static]`.
//
// As '0 gets outlived by 'a - its "upper bound" -  the only applicable choice
// region is 'a.
//
// '1 has to outlive 'b so the only applicable choice regions are 'b and 'static.
// Considering this member constraint by itself would choose 'b as it is the
// smaller of the two regions.
//
// However, this is only the case when ignoring the member constraint on '0.
// After applying this constraint and requiring '0 to outlive 'a. As '1 outlives
// '0, the region 'b is no longer an applicable choice region for '1 as 'b does
// does not outlive 'a. We would therefore choose 'static.
//
// This means applying member constraints is order dependent. We handle this by
// first applying member constraints for regions 'x and then consider the resulting
// constraints when applying member constraints for regions 'y with 'y: 'x.
fn with_constraints<'r0, 'r1, 'a, 'b>() -> *mut (&'r0 (), &'r1 ())
where
    'r1: 'r0,
    'a: 'r0,
    'r1: 'b,
{
    loop {}
}
fn foo<'a, 'b>() -> impl Sized + use<'a, 'b> {
    with_constraints::<'_, '_, 'a, 'b>()
}
fn main() {}
