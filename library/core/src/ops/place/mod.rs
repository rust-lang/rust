//! # Operations on places
//!
//! This module provides traits to customize the place operations of Rust.
//!
//! ## Places and place expressions
//!
//! A *place* in Rust is a particular location in memory. Places are represented
//! by [*place expressions*][ref-place-exprs], which have the following syntax:
//!
//! [ref-place-exprs]: https://doc.rust-lang.org/reference/expressions.html#r-expr.place-value.place-expr-kinds
//!
//! - `$path`: paths that refer to local variables (also parameters) and
//!   statics,
//! - `*$place`: dereferencing another place expression,
//! - `$place[$expr]`: indexing into another place expression,
//! - `$place.$ident`: accessing a field of another place expression,
//! - `($place)`: parenthesized place expressions,
//! - `$value`: an arbitrary expression; it's value is stored in a temporary
//!   place whose lifetime is determined from its context.
//!
//! Further reading:
//! - <https://nadrieril.github.io/blog/2025/12/06/on-places-and-their-magic.html>
//! - <https://www.ralfj.de/blog/2024/08/14/places.html>
//!
//! Place expressions have a direct representation in the form of a
//! [`PlaceHandle`]. A handle points at a place and is responsible for
//! performing all available place operations on the represented place.
//!
//! ## Place Operation Traits
//!
//! Aside from the three basic place operations of [`ReadPlace`],
//! [`WritePlace`], and [`BorrowPlace`], there are also the following other
//! operations:
//!
//! - [`MovePlace`] -- moving out of a place (looks the same as a read
//!   operation).
//! - [`IndexPlace`] -- using the index operator on places (`place[idx]`).
//! - [`DerefPlace`] -- dereferencing a pointer that's in a place (`*place`).
//! - [`ProjectPlace`] -- accessing a subplace (`place.field`).
//! - [`DropPlace`] -- dropping the contents of a place (no surface syntax,
//!   emitted by the compiler).
//! - [`MetadataPlace`] -- TODO (no surface syntax, emitted by the compiler).
//!
//! The place operations are desugared in a similar manner to place expressions.
//! In particular, a place operation always operates upon a place expression,
//! which relies on the desugaring detailed in the previous section. Refer to
//! the operation traits for their desugaring.
//!
//! ## Safety
//!
//! All operation functions are `unsafe`, since they have raw pointer arguments
//! that have safety preconditions. The arguments are raw pointers, because the
//! values they point to need not be in a valid state (they may be partially
//! moved out or borrowed).
//!
//! The safety requirements for the operation functions have not been figured
//! out at this point in time. Since we expect several changes to the design, we
//! do not want to commit to writing down good safety documentation before
//! having finished the design.
//!
//! What is clear at the moment is that the safety requirements will heavily
//! interact with the borrow checker. It will ensure that simultaneous place
//! operations on the same value are allowed, since they either affect disjoint
//! subplaces, or because they both only require shared access. For example:
//! - reading `ptr.field.subfield` and borrowing `ptr.field` with `&T` are
//!   allowed to happen at the same time,
//! - writing `ptr.field` and borrowing `ptr.field.subfield` at the same time is
//!   not allowed.
//!
//! The safety of using the place operations via the operators will depend on
//! the value of the `SAFE` constant in the operation traits. At the moment we
//! will only permit a literal value of `true` or `false` in implementations. It
//! will dictate if people have to write for example `unsafe { &*ptr }` or if
//! `*ptr` is allowed. It should be set to `true` when the borrow checker's
//! guarantees of either disjoint subplaces or "all concurrent operations are
//! shared" are enough to calling the operations' function correctly. If there
//! are additional requirements, such as "ptr is valid", then `SAFE` should be
//! set to `false`. For example, `&mut T` will have `SAFE = true` in
//! [`ReadPlace`], but `NonNull<T>` will set it to `false`.

mod handle;
mod wrapper;

pub use self::{
    handle::*,
    wrapper::*,
};

use crate::ptr::Pointee;

/// A type that acts as a proxy for a place.
///
/// A value of this type represents a specific place. The operations that are
/// available on that place are controlled by which *place operation traits* are
/// implemented on the handle of that place.
pub trait PlaceProxy {
    /// The type stored at the proxied place.
    type Target: ?Sized;
}

/// A type that acts as a proxy for places indexed by `Idx`.
///
/// A value of this type represents a variable number of places. The operations
/// that are available on those places are controlled by which *place operation
/// traits* are implemented on the handle of that place.
pub trait IndexProxy<Idx> {
    /// The type stored at each proxied place.
    type Element: ?Sized;
}

/// Read from a place.
pub unsafe trait ReadPlace: PlaceHandle {
    /// Read from the place.
    unsafe fn read_place(self) -> Self::Target;
}

/// Move out of a place.
///
/// When reading from a place, the type of the contents of the place must
/// implement [`Copy`], or the handle must implement this trait. This trait
/// allows moving out of the place, leaving it in a partially initialized state.
///
/// When implementing this trait, [`DropPlace`] should almost always also be
/// implemented, since otherwise dropping a partially moved-out proxy is not
/// permitted. Additionally, the proxy should implement [`DropHusk`] for the
/// same reason.
pub unsafe trait MovePlace: ReadPlace {}

/// Write to a place.
pub unsafe trait WritePlace: PlaceHandle {
    /// Write to the place.
    unsafe fn write_place(self, value: Self::Target);
}

/// Borrow a place as `Output`.
pub unsafe trait BorrowPlace<Output>: PlaceHandle {
    /// Borrow the place as `Output`.
    unsafe fn borrow_place(self) -> Output;
}

pub unsafe trait MetadataPlace: PlaceHandle {
    unsafe fn metadata(self) -> <Self::Target as Pointee>::Metadata;
}

/// Drop the contents of a place.
///
/// This operation should only drop the value at the place and not invalidate
/// the proxy itself. A new value might be moved back in later with
/// [`WritePlace`].
pub unsafe trait DropPlace: PlaceHandle {
    unsafe fn drop_place(self);
}

/// Destroy a [`PlaceProxy`] where all contents have been moved out/dropped.
///
/// This is essentially like [`Drop`], but supports the situation where the
/// value stored in the place represented by `Self` have been moved out,
/// dropped, or never initialized to begin with.
pub unsafe trait DropHusk: PlaceProxy {
    /// Destroy the proxy at `*this`.
    // TODO: do not use raw pointer here! (but what instead?)
    unsafe fn drop_husk(this: *mut Self);
}
