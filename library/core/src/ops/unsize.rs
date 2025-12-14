use crate::marker::{PointeeSized, Unsize};

/// Trait that indicates that this is a pointer or a wrapper for one,
/// where unsizing can be performed on the pointee.
///
/// See the [DST coercion RFC][dst-coerce] and [the nomicon entry on coercion][nomicon-coerce]
/// for more details.
///
/// For builtin pointer types, pointers to `T` will coerce to pointers to `U` if `T: Unsize<U>`
/// by converting from a thin pointer to a fat pointer.
///
/// For custom types, the coercion here works by coercing `Foo<T>` to `Foo<U>`
/// provided an impl of `CoerceUnsized<Foo<U>> for Foo<T>` exists.
/// Such an impl can only be written if `Foo<T>` has only a single non-phantomdata
/// field involving `T`. If the type of that field is `Bar<T>`, an implementation
/// of `CoerceUnsized<Bar<U>> for Bar<T>` must exist. The coercion will work by
/// coercing the `Bar<T>` field into `Bar<U>` and filling in the rest of the fields
/// from `Foo<T>` to create a `Foo<U>`. This will effectively drill down to a pointer
/// field and coerce that.
///
/// Generally, for smart pointers you will implement
/// `CoerceUnsized<Ptr<U>> for Ptr<T> where T: Unsize<U>, U: ?Sized`, with an
/// optional `?Sized` bound on `T` itself. For wrapper types that directly embed `T`
/// like `Cell<T>` and `RefCell<T>`, you
/// can directly implement `CoerceUnsized<Wrap<U>> for Wrap<T> where T: CoerceUnsized<U>`.
/// This will let coercions of types like `Cell<Box<T>>` work.
///
/// [`Unsize`][unsize] is used to mark types which can be coerced to DSTs if behind
/// pointers. It is implemented automatically by the compiler.
///
/// [dst-coerce]: https://github.com/rust-lang/rfcs/blob/master/text/0982-dst-coercion.md
/// [unsize]: crate::marker::Unsize
/// [nomicon-coerce]: ../../nomicon/coercions.html
#[unstable(feature = "coerce_unsized", issue = "18598")]
#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T: PointeeSized> {
    // Empty.
}

// &mut T -> &mut U
#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<&'a mut U> for &'a mut T {}
// &mut T -> &U
#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<'a, 'b: 'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<&'a U> for &'b mut T {}
// &mut T -> *mut U
#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*mut U> for &'a mut T {}
// &mut T -> *const U
#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*const U> for &'a mut T {}

// &T -> &U
#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<'a, 'b: 'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<&'a U> for &'b T {}
// &T -> *const U
#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*const U> for &'a T {}

// *mut T -> *mut U
#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*mut U> for *mut T {}
// *mut T -> *const U
#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*const U> for *mut T {}

// *const T -> *const U
#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*const U> for *const T {}

/// `DispatchFromDyn` is used in the implementation of dyn-compatibility[^1] checks (specifically
/// allowing arbitrary self types), to guarantee that a method's receiver type can be dispatched on.
///
/// *Note*: `DispatchFromDyn` was briefly named `CoerceSized` which had a slightly different
/// interpretation.
///
/// Imagine we have a trait object `t` with type `&dyn Tr`, where `Tr` is some trait with a method
/// `m` defined as `fn m(&self);`.
/// When calling `t.m()`, the receiver `t` is a wide pointer, but an implementation of `m` will
/// expect a narrow pointer as `&self`, specifically a reference to the concrete type.
/// The compiler must generate an implicit conversion from the trait object `&dyn Trait` or
/// wide pointer `&UnsizedType` to a concrete reference `&ConcreteType` or narrow pointer `&SizedType`
/// respectively.
///
/// Implementing `DispatchFromDyn` indicates that such conversion is allowed and, thus, that the
/// type implementing `DispatchFromDyn` is safe to use as the type of `self`, also known as a method
/// receiver type, in an dyn-compatible method.
/// In the above example, the compiler will require `DispatchFromDyn` is implemented for `&'a T`
/// against `&'a dyn Tr`, given any `T` with `T: Unsize<dyn Tr>`.
///
/// `DispatchFromDyn` does not specify the conversion from wide pointer to narrow pointer.
/// The conversion is *hard-wired* into the compiler.
/// Therefore, the compiler will check that the following properties must hold,
/// so that it is only safe to implement `DispatchFromDyn` for types with these properties.
///
/// * *Either* `Self := &A` and `T := &B` are either both references or both raw pointers to
///   generic types `A` and `B`, so that `A: Unsize<B>`[^2].
///   In addition, the mutability of the references or raw pointers must match.
///   In other words, `&A`/`&B` and `&mut A`/`&mut B` are valid pairings.
/// * *Or*, all of the following hold:
///   - `Self` and `T` must have the same type constructor, and only vary in a single type parameter
///     formal which is undergoing *coercion*.
///     For instance, `impl<T: Unsize<T> + PointeeSized, U + PointeeSized> DispatchFromDyn<Rc<U>> for Rc<T>`
///     is acceptable because the single type parameter, the `T` and `U` respectively, is the coerced type.
///     `impl<T: Unsize<T> + PointeeSized, U + PointeeSized> DispatchFromDyn<Arc<U>> for Rc<T>` is
///     unacceptable because `Arc<U>` and `Rc<T>` does not match on the type constructor.
///     One is `Arc<_>` and the other is `Rc<_>`.
///   - The definition for `Self` must be a `struct`.
///   - The definition for `Self` must not be `#[repr(packed)]` or `#[repr(C)]`.
///   - Excluding one-aligned zero-sized fields, the definition for `Self` must have exactly one
///     field and that field's type must be of the coerced type.
///     Furthermore, the type `FSelf` of this cocerced field type in `Self` must also implement
///     `DispatchFromDyn<FTarget>` where `FTarget` is the type of the corresponding coerced field in
///     `T`.
///
/// Note that we do not support the case where multiple pointer narrowings or downcasting of trait
/// objects into concrete objects are involved in coercing one type to another through
/// the `DispatchFromDyn` trait.
/// For instance, we do not support dispatch from `&Arc<dyn Tr>` to `&Arc<T>`.
/// `&Arc<dyn Tr>` is already behind an immutable reference while the coercion would require a
/// downcasting of the trait object `Arc<dyn Tr>` by chipping of the virtual table.
/// This will break the invariant of an immutable reference that the place behind shall remain
/// not mutated during the access.
/// Similarly, the dispatch from `Box<Box<dyn Tr>>` to `Box<Box<T>>` also requires a similar memory
/// layout change.
/// Therefore, this category of coercion is not supported.
///
/// An example implementation of the trait:
///
/// ```
/// # #![feature(dispatch_from_dyn, unsize)]
/// # use std::{ops::DispatchFromDyn, marker::Unsize};
/// # struct Rc<T: ?Sized>(std::rc::Rc<T>);
/// impl<T: ?Sized, U: ?Sized> DispatchFromDyn<Rc<U>> for Rc<T>
/// where
///     T: Unsize<U>,
/// {}
/// ```
///
/// [^1]: Formerly known as *object safety*.
/// [^2]: crate::marker::Unsize
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
#[lang = "dispatch_from_dyn"]
pub trait DispatchFromDyn<T> {
    // Empty.
}

// &T -> &U
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<&'a U> for &'a T {}
// &mut T -> &mut U
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<&'a mut U> for &'a mut T {}
// *const T -> *const U
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<*const U> for *const T {}
// *mut T -> *mut U
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<*mut U> for *mut T {}
