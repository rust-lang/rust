//! Operations on places.
//!
//! # Operations on Places
//!
//! This module contains traits to customize the place operations of Rust:
//! - reading,
//! - writing, and
//! - borrowing.
//!
//! This is part of the language experiment for field projections
//! <https://github.com/rust-lang/rust/issues/145383>. The specific design that
//! is currently being implemented is explained in detail in the latest [design
//! meeting document](https://hackmd.io/H5d2-83ER2ymNPZVIWCYWg?view). Note that
//! several types and traits have been renamed. Further modifications pending
//! experiment results are expected to occur.
//!
//! ## Places
//!
//! A *place* in Rust is a particular location in memory. They are represented
//! by [*place expressions*][ref-place-exprs], which take on the following form:
//! - `$path`: paths that refer to locals variables (also parameters) and
//!   statics,
//! - `*$place`: dereferences of another place expression,
//! - `$place[$expr]`: indexing operation of another place expression,
//! - `$place.$ident`: field access of another place expression,
//! - `($place)`: parenthesized place expressions,
//! - `$value`: value expressions can be coerced to a temporary place whose
//!   lifetime is determined from its context,
//!
//! Further reading:
//! - <https://nadrieril.github.io/blog/2025/12/06/on-places-and-their-magic.html>
//! - <https://www.ralfj.de/blog/2024/08/14/places.html>
//!
//! [ref-place-exprs]: https://doc.rust-lang.org/reference/expressions.html#r-expr.place-value.place-expr-kinds
//!
//! ## Place Proxies
//!
//! Places and place expressions are not part of the type system of Rust and
//! therefore cannot implement traits. To still be able to customize place
//! operations via traits, they are implemented for types that act as a proxy
//! for the place. Such types are called *place proxies* and they implement the
//! [`PlaceProxy`] trait. Elements of a place proxy type denote/represent one
//! specific place. Such an element can be dereferenced, which results in a
//! place expression denoting the represented place. This is a normal place
//! expression that can be used to perform place operations or can be composed
//! into a bigger place expression.
//!
//! Place operations performed on the represented place are implemented by the
//! corresponding place operation trait of this module. This trait must be
//! implemented for the place's proxy type, otherwise the compiler will emit an
//! error:
//!
//! - reading [`ReadPlace`],
//! - writing [`WritePlace`], and
//! - borrowing [`BorrowPlace`].
//!
//! Examples of types that implement some of these operations are smart & dumb
//! pointers like `Box<T>`, `Arc<T>`, [`&mut T`](primitive@reference),
//! [`*mut T`](primitive@pointer), and [`NonNull<T>`](core::ptr::NonNull).
//!
//! ### Subplaces
//!
//! Place operations are allowed to target only a *subplace*. For example
//! `my_struct.field = 42;` writes to only the `field` subplace. Since places
//! are operated upon through their proxy, which represents the *entire* place,
//! each operation has a generic parameter denoting the subplace the operation
//! is targeting. This generic implements the [`Subplace`] trait, which contains
//! all the information describing the targeted subplace.
//!
//! Note that the [`Subplace`] trait also can represent the entire place in case
//! the operation affects the entire place (for example `let x = *ptr;`).
//!
//! This generic argument is supplied by the compiler when desugaring a place
//! operation into the corresponding place operation trait function call. There
//! is a direct translation from place expressions to types implementing the
//! [`Subplace`] trait.
//!
//! ### Implicit Operations
//!
//! In addition to the three visible operations, there are several other
//! *implicit* operations that can be implemented for place proxies:
//!
//! - moving out of a subplace [`MovePlace`],
//! - dropping a subplace [`DropPlace`] and dropping a fully moved-out pointer
//!   [`DropHusk`], and
//! - support for accessing a nested pointer [`DerefPlace`].
//!
//! ### Place Wrappers
//!
//! Place wrappers are a special kind of place proxy. They "physically contain"
//! the place they are proxying for. A good example is [`MaybeUninit<T>`]. To
//! support subplaces of these place wrappers, the [`WrapPlace`] trait exists.
//! It allows forwarding subplaces to the proxy and changing the subplace access
//! information. With [`MaybeUninit<T>`], this allows accessing any subplace
//! under the transformation that it's type is wrapped in `MaybeUninit`. So
//! Given `&MaybeUninit<Struct>`, the `field` subplace can be borrowed using `&`
//! and it has type `&MaybeUninit<Field>`.
//!
//! [`MaybeUninit<T>`]: crate::mem::MaybeUninit
//!
//! ### Safety
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

use crate::ptr::Pointee;

/// A subplace of [`Self::Source`] with the type [`Self::Target`].
///
/// A subplace is always within the same allocation as the base place. A
/// subplace is described by a chain of
/// - field accesses, and
/// - array/slice indexes.
///
/// Note that a subplace can also be the entire original place.
///
/// # Safety
///
/// See the module-level section on [safety](crate::ops::place#safety).
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

/// Marks a type as a place proxy.
///
/// A place proxy can be dereferenced (`*val`). This results in a place for
/// which any place operation is implemented by this type. The operation is
/// available only if the corresponding place operation trait is implemented:
///
/// - [`ReadPlace`]
/// - [`WritePlace`]
/// - [`BorrowPlace`]
///
/// Furthermore, there are implicit place operations that can be supported by
/// types implementing this trait:
///
/// - [`MovePlace`]
/// - [`DropPlace`]
/// - [`DropHusk`]
/// - [`DerefPlace`]
/// - [`WrapPlace`]
///
/// Read the [module](self) description for more information.
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "place_proxy"]
pub trait PlaceProxy {
    /// The type of the contained place.
    #[lang = "place_proxy_target"]
    type Target: ?Sized;
}

/// Reading a place `let val = *x;`.
///
/// When `x: Self`, then `let val = *x;` will be desugared into [`ReadPlace::read`].
///
/// # Safety
///
/// See the module-level section on [safety](crate::ops::place#safety).
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "read_place"]
pub unsafe trait ReadPlace<S>: PlaceProxy
where
    S: Subplace<Source = Self::Target>,
    S::Target: Sized,
{
    /// Whether the read operation is safe when used through the operator.
    ///
    /// When the operator is used, the borrow checker follows its usual rules to
    /// ensure that no other operation conflicts with this one. If that alone is
    /// sufficient to make this operation sound, then this should be `true`.
    #[lang = "read_place_safe"]
    const SAFE: bool;

    /// Reads the subplace pointed to by `this`.
    ///
    /// # Safety
    ///
    /// See the module-level section on [safety](crate::ops::place#safety).
    #[lang = "read_place_read"]
    unsafe fn read(this: *const Self, sub: S) -> S::Target;
}

/// Writing a place `*x = val;`.
///
/// When `x: Self`, then `*x = val;` will be desugared into
/// [`WritePlace::write`].
///
/// When a value already exists at the subplace, it is dropped with
/// [`DropPlace`] before it is written to.
///
/// # Safety
///
/// See the module-level section on [safety](crate::ops::place#safety).
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "write_place"]
pub unsafe trait WritePlace<S>: PlaceProxy
where
    S: Subplace<Source = Self::Target>,
    S::Target: Sized,
{
    /// Whether the write operation is safe when used through the operator.
    ///
    /// When the operator is used, the borrow checker follows its usual rules to
    /// ensure that no other operation conflicts with this one. If that alone is
    /// sufficient to make this operation sound, then this should be `true`.
    #[lang = "write_place_safe"]
    const SAFE: bool;

    /// Writes to the subplace pointed to by `this`.
    ///
    /// # Safety
    ///
    /// See the module-level section on [safety](crate::ops::place#safety).
    #[lang = "write_place_write"]
    unsafe fn write(this: *const Self, sub: S, value: S::Target);
}

/// Borrowing a place with `X`.
///
/// When `y: Self`, then `let x = @<X> *y;` will be desugared into
/// [`BorrowPlace::borrow`].
///
/// # Safety
///
/// See the module-level section on [safety](crate::ops::place#safety).
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "borrow_place"]
pub unsafe trait BorrowPlace<S, X>: PlaceProxy
where
    S: Subplace<Source = Self::Target>,
    X: PlaceProxy<Target = S::Target>,
{
    /// Whether the borrow operation is safe when used through the operator.
    ///
    /// When the operator is used, the borrow checker follows its usual rules to
    /// ensure that no other operation conflicts with this one. If that alone is
    /// sufficient to make this operation sound, then this should be `true`.
    #[lang = "borrow_place_safe"]
    const SAFE: bool;

    // FIXME: this is missing some associated items related to controlling the
    // borrow checker. The details need to still be worked out in a-mir-formality.

    /// Borrow the subplace pointed to by `this` with `X`.
    ///
    /// # Safety
    ///
    /// See the module-level section on [safety](crate::ops::place#safety).
    #[lang = "borrow_place_borrow"]
    unsafe fn borrow(this: *const Self, sub: S) -> X;
}

/// Moving out of a place.
///
/// When `x: Self` and one performs a [`ReadPlace::read`] where the target value
/// is not [`Copy`], then the compiler checks if this trait is implemented and
/// if so, moves the value out by reading it and adjusting the borrow checker
/// state of the place.
///
/// # Safety
///
/// See the module-level section on [safety](crate::ops::place#safety).
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
/// Emitted by the compiler before a new value is written ([`WritePlace`]) to
/// this subplace, or when a pointer to a partially moved out place is dropped.
/// See the documentation of [`DropHusk`] for more on the exact details of that
/// last case.
///
/// # Safety
///
/// See the module-level section on [safety](crate::ops::place#safety).
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "drop_place"]
pub unsafe trait DropPlace<S>: PlaceProxy
where
    S: Subplace<Source = Self::Target>,
{
    /// Drop the subplace pointed to by `this`.
    ///
    /// # Safety
    ///
    /// See the module-level section on [safety](crate::ops::place#safety).
    #[lang = "drop_place_drop"]
    unsafe fn drop(this: *const Self, sub: S);
}

/// Dropping an empty place proxy.
///
/// This operation is emitted by the compiler when a place proxy is being
/// dropped where all fields of the represented place were moved out.
///
/// If no fields or only some fields have been moved out, all not yet moved out
/// fields are dropped with the [`DropPlace`] trait. After that this pointer is
/// dropped by calling [`DropHusk::drop_husk`].
///
/// Note that a write operation ([`WritePlace`]) can move a value into a
/// previously moved-out field.
///
/// This trait cannot be implemented at the same time as [`Drop`], since it
/// generalizes it. When this trait is implemented, dropping a value of type
/// `Self` is done by first dropping the represented place using [`DropPlace`]
/// and then calling drop_husk.
///
/// # Safety
///
/// See the module-level section on [safety](crate::ops::place#safety).
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "drop_husk"]
pub unsafe trait DropHusk: PlaceProxy {
    /// Drops the
    ///
    /// # Safety
    ///
    /// See the module-level section on [safety](crate::ops::place#safety).
    #[lang = "drop_husk_drop_husk"]
    unsafe fn drop_husk(this: *const Self);
}

/// Accessing a nested proxy.
///
/// When `x: Self`, then nested dereferences `let _ = **x;` are desugared into a
/// combination of the corresponding operation and [`DerefPlace::deref`].
///
/// # Safety
///
/// See the module-level section on [safety](crate::ops::place#safety).
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "deref_place"]
pub unsafe trait DerefPlace<S>: PlaceProxy
where
    S: Subplace<Source = Self::Target>,
    S::Target: PlaceProxy,
{
    /// Obtain a raw pointer to the subplace contained by `this`.
    ///
    /// # Safety
    ///
    /// See the module-level section on [safety](crate::ops::place#safety).
    #[lang = "deref_place_deref"]
    unsafe fn deref(this: *const Self, sub: S) -> *const S::Target;
}

/// Forwards the subplace `S` of the place contained by this.
///
/// When `x: Self` and `Self::Target` has a subplace `S` accessible via
/// `.foo.bar`, then `x.foo.bar` is also valid, has type `<Self::Wrapped as
/// Subplace>::Target` and any place operation on it uses
/// <code>[Self::wrap]\(sub\)</code> as the subplace instead of `sub`.
///
/// # Safety
///
/// See the module-level section on [safety](crate::ops::place#safety).
#[unstable(feature = "field_projections", issue = "145383")]
#[lang = "wrap_place"]
pub unsafe trait WrapPlace<S>: PlaceProxy
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
