// A test checking whether we eagerly apply subtype requirements
// when relating types. This relies on incomplete inference with
// higher-ranked types. If we eagerly apply the `?x <: ?y` subtype
// requirements after constraining `?x` to `for<'a> fn(&'a ())` we
// would incorrectly constrain `?y` to also be `for<'a> fn(&'a ())`.
//
// This would then result in an error when relating `y` with `Inv<fn(&'static ())>`.
//
// It's fine for this behavior to change, we should do so intentionally however.

//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

fn to_inv<T>(x: Option<T>) -> Inv<T> {
    Inv(None)
}
#[derive(Copy, Clone)]
struct Inv<T>(Option<*mut T>);

fn main() {
    let x = None;
    let y = x;
    let mut x = to_inv(x);
    let y = to_inv(y);
    // deferred ?x <: ?y
    let z = Inv(None);
    // type_of(w) = (Inv<?x>, Inv<?z>, Inv<?z>)
    let mut w = (x, z, z);
    // type_of(a) = (Inv<for<'a> fn(&'a ())>, Inv<?y>, Inv<fn(&'static ())>)
    let a = (Inv::<fn(&())>(None), y, Inv::<fn(&'static ())>(None));
    // ?x = for<'a> fn(&'a ())
    // ?y = fn(&'static ())
    w = a;
}
