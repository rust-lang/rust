//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// The next solver can create extra uses of these opaques with fresh non-captured
// parent regions. They are still the same opaque instantiation, but their hidden
// regions may be unconstrained.
//
// If member constraints run before those candidates are related, an unconstrained
// region can be assigned a valid but wrong member such as `'static`.
fn new_defining_use<F: FnOnce(T) -> R, T, R>(_: F) {}

fn rpit1<'a, 'b: 'b>(x: &'b ()) -> impl Sized + use<'a, 'b> {
    new_defining_use(rpit1::<'a, 'b>);
    x
}

struct Inv<'a, 'b>(*mut (&'a (), &'b ()));
fn rpit2<'a>(_: ()) -> impl Sized + use<'a> {
    new_defining_use(rpit2::<'a>);
    Inv::<'a, 'static>(std::ptr::null_mut())
}
fn main() {}
