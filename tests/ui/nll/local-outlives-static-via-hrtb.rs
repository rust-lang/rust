// Test that we handle the case when a local variable is borrowed for `'static`
// due to an outlives constraint involving a region in an incompatible universe

pub trait Outlives<'this> {}

impl<'this, T> Outlives<'this> for T where T: 'this {}
trait Reference {
    type AssociatedType;
}

impl<'a, T: 'a> Reference for &'a T {
    type AssociatedType = &'a ();
}

fn assert_static_via_hrtb<G>(_: G) where for<'a> G: Outlives<'a> {}

fn assert_static_via_hrtb_with_assoc_type<T>(_: &'_ T)
where
    for<'a> &'a T: Reference<AssociatedType = &'a ()>,
{}

fn main() {
    let local = 0;
    assert_static_via_hrtb(&local); //~ ERROR `local` does not live long enough
    assert_static_via_hrtb_with_assoc_type(&&local); //~ ERROR `local` does not live long enough
}
