//@ check-pass
//@ compile-flags: -Znext-solver

// Regression test for trait-system-refactor-initiative#264.
//
// Some defining uses of opaque types can't constrain captured regions to universals.
// Previouly, we eagerly report error in this case.
// Now we report error only if there's no fully defining use from all bodies of the typeck root.

struct Inv<'a>(*mut &'a ());

fn mk_static() -> Inv<'static> { todo!() }

fn guide_closure_sig<'a>(f: impl FnOnce() -> Inv<'a>) {}

fn unconstrained_in_closure() -> impl Sized {
    guide_closure_sig(|| unconstrained_in_closure());
    mk_static()
}

fn main() {}
