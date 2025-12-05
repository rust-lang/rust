//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Proj<'a> {
    type Assoc;
}

impl<'a, 'b, F: FnOnce() -> &'b ()> Proj<'a> for F {
    type Assoc = ();
}

fn is_proj<F: for<'a> Proj<'a>>(f: F) {}

fn define<'a>() -> impl Sized + use<'a> {
    // This defines the RPIT to `&'unconstrained_b ()`, an inference
    // variable which is in a higher universe as gets created inside
    // of the binder of `F: for<'a> Proj<'a>`. This previously caused
    // us to not apply member constraints.
    //
    // This was unnecessary. It is totally acceptable for member regions
    // to be able to name placeholders from higher universes, as long as
    // they don't actually do so.
    is_proj(define::<'a>);
    &()
}

fn main() {}
