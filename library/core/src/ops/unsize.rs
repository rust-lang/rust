use crate::marker::Unsize;

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
#[unstable(feature = "coerce_unsized", issue = "27732")]
#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T: ?Sized> {
    // Empty.
}

// &mut T -> &mut U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a mut U> for &'a mut T {}
// &mut T -> &U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b mut T {}
// &mut T -> *mut U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for &'a mut T {}
// &mut T -> *const U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for &'a mut T {}

// &T -> &U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
// &T -> *const U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for &'a T {}

// *mut T -> *mut U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}
// *mut T -> *const U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *mut T {}

// *const T -> *const U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *const T {}

/// `DispatchFromDyn` is used in the implementation of object safety checks (specifically allowing
/// arbitrary self types), to guarantee that a method's receiver type can be dispatched on.
///
/// Note: `DispatchFromDyn` was briefly named `CoerceSized` (and had a slightly different
/// interpretation).
///
/// Imagine we have a trait object `t` with type `&dyn Tr`, where `Tr` is some trait with a method
/// `m` defined as `fn m(&self);`. When calling `t.m()`, the receiver `t` is a wide pointer, but an
/// implementation of `m` will expect a narrow pointer as `&self` (a reference to the concrete
/// type). The compiler must generate an implicit conversion from the trait object/wide pointer to
/// the concrete reference/narrow pointer. Implementing `DispatchFromDyn` indicates that that
/// conversion is allowed and thus that the type implementing `DispatchFromDyn` is safe to use as
/// the self type in an object-safe method. (in the above example, the compiler will require
/// `DispatchFromDyn` is implemented for `&'a U`).
///
/// `DispatchFromDyn` does not specify the conversion from wide pointer to narrow pointer; the
/// conversion is hard-wired into the compiler. For the conversion to work, the following
/// properties must hold (i.e., it is only safe to implement `DispatchFromDyn` for types which have
/// these properties, these are also checked by the compiler):
///
/// * EITHER `Self` and `T` are either both references or both raw pointers; in either case, with
///   the same mutability.
/// * OR, all of the following hold
///   - `Self` and `T` must have the same type constructor, and only vary in a single type parameter
///     formal (the *coerced type*, e.g., `impl DispatchFromDyn<Rc<T>> for Rc<U>` is ok and the
///     single type parameter (instantiated with `T` or `U`) is the coerced type,
///     `impl DispatchFromDyn<Arc<T>> for Rc<U>` is not ok).
///   - The definition for `Self` must be a struct.
///   - The definition for `Self` must not be `#[repr(packed)]` or `#[repr(C)]`.
///   - Other than one-aligned, zero-sized fields, the definition for `Self` must have exactly one
///     field and that field's type must be the coerced type. Furthermore, `Self`'s field type must
///     implement `DispatchFromDyn<F>` where `F` is the type of `T`'s field type.
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
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
#[lang = "dispatch_from_dyn"]
pub trait DispatchFromDyn<T> {
    // Empty.
}

// &T -> &U
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<&'a U> for &'a T {}
// &mut T -> &mut U
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<&'a mut U> for &'a mut T {}
// *const T -> *const U
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<*const U> for *const T {}
// *mut T -> *mut U
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<*mut U> for *mut T {}
