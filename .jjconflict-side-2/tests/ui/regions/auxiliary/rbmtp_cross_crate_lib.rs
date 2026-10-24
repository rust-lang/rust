// Check that method bounds declared on traits/impls in a cross-crate
// scenario work. This is the library portion of the test.

pub enum MaybeOwned<'a> {
    Owned(isize),
    Borrowed(&'a isize)
}

pub struct Inv<'a> { // invariant w/r/t 'a
    x: &'a mut &'a isize
}

// I encountered a bug at some point with encoding the IntoMaybeOwned
// trait, so I'll use that as the template for this test.
pub trait IntoMaybeOwned<'a> {
    fn into_maybe_owned(self) -> MaybeOwned<'a>;

    // Note: without this `into_inv` method, the trait is
    // contravariant w/r/t `'a`, since if you look strictly at the
    // interface, it only returns `'a`. This complicates the
    // downstream test since it wants invariance to force an error.
    // Hence we add this method.
    fn into_inv(self) -> Inv<'a>;

    fn bigger_region<'b:'a>(self, b: Inv<'b>);
}

impl<'a> IntoMaybeOwned<'a> for Inv<'a> {
    fn into_maybe_owned(self) -> MaybeOwned<'a> { panic!() }
    fn into_inv(self) -> Inv<'a> { panic!() }
    fn bigger_region<'b:'a>(self, b: Inv<'b>) { panic!() }
}
