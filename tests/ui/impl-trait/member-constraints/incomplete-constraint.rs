//@ check-pass
// FIXME(-Znext-solver): enable this test

// These functions currently do not normalize the opaque type but will do
// so in the future. At this point we've got a new use of the opaque with fully
// universal arguments but for which lifetimes in the hidden type are unconstrained.
//
// Applying the member constraints would then incompletely infer `'unconstrained` to `'static`.
fn new_defining_use<F: FnOnce(T) -> R, T, R>(_: F) {}

fn rpit1<'a,  'b: 'b>(x: &'b ()) -> impl Sized + use<'a, 'b> {
    new_defining_use(rpit1::<'a, 'b>);
    x
}

struct Inv<'a, 'b>(*mut (&'a (), &'b ()));
fn rpit2<'a>(_: ()) -> impl Sized + use<'a> {
    new_defining_use(rpit2::<'a>);
    Inv::<'a, 'static>(std::ptr::null_mut())
}
fn main() {}
