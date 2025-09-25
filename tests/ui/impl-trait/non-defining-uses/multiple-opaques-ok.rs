//@ compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass
#![allow(unconditional_recursion)]

// Regression test for trait-system-refactor-initiative#18, making sure
// we support being sub unified with more than 1 opaque type.

trait Id {
    type This;
}
impl Id for &'static str {
    type This = &'static str;
}
fn to_assoc<T: Id>(x: T) -> <T as Id>::This {
    todo!()
}

fn mirror1() -> (impl Id<This = &'static str>, impl Sized) {
    let mut opaque = mirror1().0;
    opaque = mirror1().1;
    let x = to_assoc(opaque);
    // `?x` equals both opaques, make sure we still use the applicable
    // item bound.
    x.len();
    (x, x)
}
fn mirror2() -> (impl Sized, impl Id<This = &'static str>) {
    let mut opaque = mirror2().0;
    opaque = mirror2().1;
    let x = to_assoc(opaque);
    // `?x` equals both opaques, make sure we still use the applicable
    // item bound.
    x.len();
    (x, x)
}
fn main() {}
