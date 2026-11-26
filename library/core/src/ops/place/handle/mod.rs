//! # Handles to Places
//!
//! The information a handle encodes can be broken into three primary categories:
//!
//! - The type of the place the handle refers to.
//! - Runtime information needed to access the place; usually, a pointer.
//! - The capabilities of the pointer the handle was derived from.

use super::{IndexProxy, PlaceProxy};
use crate::field::Subplace;

mod const_ptr;
mod local;
mod mut_ptr;
mod mutable_ref;
mod shared_ref;

pub use self::const_ptr::PtrHandle;
pub use self::local::LocalHandle;
pub use self::mut_ptr::MutPtrHandle;
pub use self::mutable_ref::MutHandle;
pub use self::shared_ref::RefHandle;

/// A *handle* to a place.
///
/// All place operations are carried out by handles. Whether a place operation
/// is available depends on the handle type implementing the respective [*place
/// operation trait*][self#place-operation-traits].
///
/// Place handles are usually not visible in user-written code. Instead, they
/// and the use of their operations are emitted by the compiler as part of
/// desugaring [*place expressions*](self#places-and-place-expressions) and
/// place operations.
///
/// Types implementing [`PlaceHandle`] do not carry *borrow checker invariants*,
/// like uniqueness of ownership or access. Instead they may only rely on the
/// local invariants granted by the place operation traits (e.g. [`WritePlace`]
/// grants unique access for the write).
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
