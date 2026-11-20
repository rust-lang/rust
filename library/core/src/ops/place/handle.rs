//! # Handles to Places
//!
//! The information a handle encodes can be broken into three primary categories:
//!
//! - The type of the place the handle refers to.
//! - Runtime information needed to access the place; usually, a pointer.
//! - The capabilities of the pointer the handle was derived from.

use super::{PlaceProxy, IndexProxy};
use crate::field::Subplace;

/// A *handle* to a place.
///
/// All place operations are carried out by handles. Whether a place operation
/// is available depends on the handle type implementing the respective [*place
/// operation trait*][self#place-operation-traits].
///
/// Place handles are not user-facing. Instead, they and the use of their
/// operations are emitted by the compiler as part of desugaring [*place
/// expressions*](self#places-and-place-expressions) and place operations.
pub trait PlaceHandle: Sized {
    /// The type stored at the place.
    type Target: ?Sized;
}

/// Dereference the proxy stored in this place.
pub unsafe trait DerefPlace: PlaceHandle
where
    Self::Target: PlaceProxy,
{
    type Derefed: PlaceHandle<Target = <Self::Target as PlaceProxy>::Target>;

    unsafe fn deref_place(self) -> Self::Derefed;
}

/// Index the proxy stored in this place.
pub unsafe trait IndexPlace<Idx>: PlaceHandle
where
    Self::Target: IndexProxy<Idx>,
{
    type Indexed: PlaceHandle<Target = <Self::Target as IndexProxy<Idx>>::Element>;

    unsafe fn index_place(self, idx: Idx) -> Self::Indexed;
}

/// Project a handle to a subplace (e.g. `place.field`).
pub unsafe trait ProjectPlace<S>: PlaceHandle
where
    S: Subplace<Source = Self::Target>,
{
    type Projected: PlaceHandle<Target = S::Target>;

    unsafe fn project_place(self, subplace: S) -> Self::Projected;
}

pub struct LocalHandle<T>(*mut T);
