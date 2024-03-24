//! A module for working with referenced data.

/// A trait for borrowing data from an owned struct
pub trait OwnedToRef {
    /// The resulting type referencing back to Self
    type Borrowed<'a>
    where
        Self: 'a;

    /// Creates a new object referencing back to the self for storage
    fn owned_to_ref(&self) -> Self::Borrowed<'_>;
}

/// A trait for cloning a referenced structure and getting owned objects
///
/// This is the pendant to [`OwnedToRef`]
pub trait RefToOwned<'a> {
    /// The resulting type after obtaining ownership.
    type Owned: OwnedToRef<Borrowed<'a> = Self>
    where
        Self: 'a;

    /// Creates a new object taking ownership of the data
    fn ref_to_owned(&self) -> Self::Owned;
}

impl<T> OwnedToRef for Option<T>
where
    T: OwnedToRef,
{
    type Borrowed<'a> = Option<T::Borrowed<'a>> where T: 'a;

    fn owned_to_ref(&self) -> Self::Borrowed<'_> {
        self.as_ref().map(|o| o.owned_to_ref())
    }
}

impl<'a, T> RefToOwned<'a> for Option<T>
where
    T: RefToOwned<'a> + 'a,
    T::Owned: OwnedToRef,
{
    type Owned = Option<T::Owned>;
    fn ref_to_owned(&self) -> Self::Owned {
        self.as_ref().map(|o| o.ref_to_owned())
    }
}

#[cfg(feature = "alloc")]
mod allocating {
    use super::{OwnedToRef, RefToOwned};
    use alloc::boxed::Box;

    impl<'a> RefToOwned<'a> for &'a [u8] {
        type Owned = Box<[u8]>;

        fn ref_to_owned(&self) -> Self::Owned {
            Box::from(*self)
        }
    }

    impl OwnedToRef for Box<[u8]> {
        type Borrowed<'a> = &'a [u8];

        fn owned_to_ref(&self) -> Self::Borrowed<'_> {
            self.as_ref()
        }
    }
}
