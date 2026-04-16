//! Place Operations.

use crate::ptr::Pointee;

/// A subplace of [`Self::Source`] with the type [`Self::Target`].
///
/// A subplace is always within the same allocation as the base place. Current subplaces are:
/// - field accesses,
/// - array/slice indexes.
///
/// This represents an arbitrary chaining of these; it can also be empty.
///
/// # Safety
///
/// FIXME
#[unstable(feature = "field_projections", issue = "145383")]
#[rustc_deny_explicit_impl]
#[rustc_dyn_incompatible_trait]
#[lang = "subplace"]
pub unsafe trait Subplace: Sized {
    /// The type of the base place this subplace is a part of.
    #[lang = "subplace_source"]
    type Source: ?Sized;

    /// The type of this subplace.
    #[lang = "subplace_target"]
    type Target: ?Sized;

    /// The offset of this subplace.
    #[lang = "subplace_offset"]
    fn offset(
        &self,
        metadata: <Self::Source as Pointee>::Metadata,
    ) -> (usize, <Self::Target as Pointee>::Metadata);
}

/// Marks a type as containing a place.
///
/// This is the new trait for the dereference operator `*`; implementing it will allow writing `*x`
/// for `x: Self`, but not enable any operations on `*x`. For those, one of the other place
/// operation traits has to be implemented:
///
/// - [`ReadPlace`]
/// - [`WritePlace`]
/// - [`BorrowPlace`]
/// - [`DropPlace`]
/// - [`MovePlace`]
/// - [`NestPlace`]
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "deref_place"]
pub trait DerefPlace {
    /// The type of the contained place.
    #[lang = "deref_place_target"]
    type Target: ?Sized;
}

/// Reading a place `let val = *x;`.
///
/// When `x: Self`, then `let val = *x;` will be desugared into [`ReadPlace::read`].
///
/// # Safety
///
/// FIXME
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "read_place"]
pub unsafe trait ReadPlace<S>: DerefPlace
where
    S: Subplace<Source = Self::Target>,
    S::Target: Sized,
{
    /// Whether the read operation is safe when used through the operator.
    ///
    /// When the operator is used, the borrow checker follows its usual rules to ensure that no
    /// other operation conflicts with this one. If that alone is sufficient to make this operation
    /// sound, then this should be `true`.
    #[lang = "read_place_safety"]
    const SAFETY: bool;

    /// Reads the subplace pointed to by `this`.
    ///
    /// # Safety
    ///
    /// FIXME
    #[lang = "read_place_read"]
    unsafe fn read(this: *const Self, sub: S) -> S::Target;
}

/// Writing a place `*x = val;`.
///
/// When `x: Self`, then `*x = val;` will be desugared into [`WritePlace::write`].
///
/// # Safety
///
/// FIXME
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "write_place"]
pub unsafe trait WritePlace<S>: DerefPlace
where
    S: Subplace<Source = Self::Target>,
    S::Target: Sized,
{
    /// Whether the write operation is safe when used through the operator.
    ///
    /// When the operator is used, the borrow checker follows its usual rules to ensure that no
    /// other operation conflicts with this one. If that alone is sufficient to make this operation
    /// sound, then this should be `true`.
    #[lang = "write_place_safety"]
    const SAFETY: bool;

    /// Writes to the subplace pointed to by `this`.
    ///
    /// # Safety
    ///
    /// FIXME
    #[lang = "write_place_write"]
    unsafe fn write(this: *const Self, sub: S, value: S::Target);
}

/// Moving out of a place.
///
/// When `x: Self` and one performs a [`ReadPlace::read`] where the target value is not [`Copy`],
/// then the compiler checks if this trait is implemented and if so, moves the value out by reading
/// it and adjusting the borrow checker state of the place.
///
/// # Safety
///
/// FIXME
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "move_place"]
pub unsafe trait MovePlace<S>: ReadPlace<S>
where
    S: Subplace<Source = Self::Target>,
    S::Target: Sized,
{
}

/// Dropping a place.
///
/// Emitted by the compiler when a place has been partially moved out and the pointer with ownership
/// is being dropped.
///
/// # Safety
///
/// FIXME
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "drop_place"]
pub unsafe trait DropPlace<S>: DerefPlace
where
    S: Subplace<Source = Self::Target>,
{
    /// Drop the subplace pointed to by `this`.
    ///
    /// # Safety
    ///
    /// FIXME
    #[lang = "drop_place_drop"]
    unsafe fn drop(this: *const Self, sub: S);
}

/// Dropping a pointer that points at a fully moved-out place.
///
/// This operation is emitted by the compiler when a pointer is being dropped that had some fields
/// moved out.
///
/// # Safety
///
/// FIXME
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "drop_husk"]
pub unsafe trait DropHusk: DerefPlace {
    /// Drops the
    ///
    /// # Safety
    ///
    /// FIXME
    #[lang = "drop_husk_drop_husk"]
    unsafe fn drop_husk(this: *const Self);
}

/// Borrowing a place with `X`.
///
/// When `y: Self`, then `let x = @<X> *y;` will be desugared into [`BorrowPlace::borrow`].
///
/// # Safety
///
/// FIXME
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "borrow_place"]
pub unsafe trait BorrowPlace<S, X>: DerefPlace
where
    S: Subplace<Source = Self::Target>,
    X: DerefPlace<Target = S::Target>,
{
    /// Whether the borrow operation is safe when used through the operator.
    ///
    /// When the operator is used, the borrow checker follows its usual rules to ensure that no
    /// other operation conflicts with this one. If that alone is sufficient to make this operation
    /// sound, then this should be `true`.
    #[lang = "borrow_place_safety"]
    const SAFETY: bool;

    // FIXME: this is missing some associated items related to controlling the
    // borrow checker. The details need to still be worked out in a-mir-formality.

    /// Borrow the subplace pointed to by `this` with `X`.
    ///
    /// # Safety
    ///
    /// FIXME
    #[lang = "borrow_place_borrow"]
    unsafe fn borrow(this: *const Self, sub: S) -> X;
}

/// Accessing a nested pointer.
///
/// When `x: Self`, then nested dereferences `let _ = **x;` is desugared into a combination of the
/// corresponding operation and a [`NestPlace::nested`].
///
/// # Safety
///
/// FIXME
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "nest_place"]
pub unsafe trait NestPlace<S>: DerefPlace
where
    S: Subplace<Source = Self::Target>,
    S::Target: DerefPlace,
{
    /// Obtain a raw pointer to the subplace contained by `this`.
    ///
    /// # Safety
    ///
    /// FIXME
    #[lang = "nest_place_nested"]
    unsafe fn nested(this: *const Self, sub: S) -> *const S::Target;
}

/// Forwards the subplace `S` of the place contained by this.
///
/// When `x: Self` and `Self::Target` has a subplace `S` accessible via `.foo.bar`, then `x.foo.bar`
/// is also valid, has type `<Self::Wrapped as Subplace>::Target` and any place operation on it uses
/// <code>[Self::wrap]\(sub\)</code> as the subplace instead of `S`.
///
/// # Safety
///
/// FIXME
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "wrap_place"]
pub unsafe trait WrapPlace<S>: DerefPlace
where
    S: Subplace<Source = Self::Target>,
{
    /// The subplace to use instead of `S` for any place operations on `Self`.
    #[lang = "wrap_place_wrapped"]
    type Wrapped: Subplace<Source = Self>;

    /// Turn a subplace of type `S` into [`Self::Wrapped`].
    #[lang = "wrap_place_wrap"]
    fn wrap(sub: S) -> Self::Wrapped;
}
