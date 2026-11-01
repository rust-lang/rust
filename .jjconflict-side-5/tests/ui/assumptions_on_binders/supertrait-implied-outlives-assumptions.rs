//@ check-pass
//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

// A trait clause implies its supertraits, so a `&'x (): Bound<'x>` requirement is also evidence
// for `&'x (): 'static`, which elaborates to the region assumption `'x: 'static`.
// `Assumptions::new` therefore takes clauses and elaborates them itself; handing it only the
// outlives clauses would drop the trait clause before it could imply anything.
//
// The clause has to mention the binder's own `'x` to survive the `max_universe == u` filter,
// while the supertrait outlives is on `'static` so that the assumption can discharge `'x: 'a`.
//
// Keeping the requirement binder-local matters: the `for<'x> Wrap<'x>: 'a` bound is proven at the
// call site below, so failing to discharge `'x: 'a` is a `NoSolution` inside the solver rather
// than a constraint escaping to the root. Constraints reaching the root are still dropped, so a
// shape which lets `'x` escape (e.g. requiring `T: 'x` for an outer `T`) would pass either way.
// Removing the `&'x (): Bound<'x>` clause below makes this fail, as does dropping trait clauses
// before elaborating.

trait Bound<'c>: 'static {}

struct Wrap<'x>(&'x ())
where
    &'x (): Bound<'x>;

fn foo<'a>(_a: &'a u32)
where
    for<'x> Wrap<'x>: 'a,
{
}

fn main() {
    foo(&10);
}
