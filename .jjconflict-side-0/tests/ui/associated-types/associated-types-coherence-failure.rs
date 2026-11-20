// Test that coherence detects overlap when some of the types in the
// impls are projections of associated type. Issue #20624.

use std::marker::PhantomData;
use std::ops::Deref;

pub struct Cow<'a, B: ?Sized>(PhantomData<(&'a (),B)>);

/// Trait for moving into a `Cow`
pub trait IntoCow<'a, B: ?Sized> {
    /// Moves `self` into `Cow`
    fn into_cow(self) -> Cow<'a, B>;
}

impl<'a, B: ?Sized> IntoCow<'a, B> for <B as ToOwned>::Owned where B: ToOwned {
    fn into_cow(self) -> Cow<'a, B> {
        Cow(PhantomData)
    }
}

impl<'a, B: ?Sized> IntoCow<'a, B> for Cow<'a, B> where B: ToOwned {
//~^ ERROR E0119
    fn into_cow(self) -> Cow<'a, B> {
        self
    }
}

impl<'a, B: ?Sized> IntoCow<'a, B> for &'a B where B: ToOwned {
//~^ ERROR E0119
    fn into_cow(self) -> Cow<'a, B> {
        Cow(PhantomData)
    }
}

impl ToOwned for u8 {
    type Owned = &'static u8;
    fn to_owned(&self) -> &'static u8 { panic!() }
}

/// A generalization of Clone to borrowed data.
pub trait ToOwned {
    type Owned;

    /// Creates owned data from borrowed data, usually by copying.
    fn to_owned(&self) -> Self::Owned;
}


fn main() {}
