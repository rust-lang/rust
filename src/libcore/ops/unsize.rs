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
/// [unsize]: ../marker/trait.Unsize.html
/// [nomicon-coerce]: ../../nomicon/coercions.html
#[unstable(feature = "coerce_unsized", issue = "27732")]
#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T: ?Sized> {
    // Empty.
}

// &mut T -> &mut U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<&'a mut U> for &'a mut T {}
// &mut T -> &U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, 'b: 'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b mut T {}
// &mut T -> *mut U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for &'a mut T {}
// &mut T -> *const U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for &'a mut T {}

// &T -> &U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, 'b: 'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
// &T -> *const U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for &'a T {}

// *mut T -> *mut U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}
// *mut T -> *const U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *mut T {}

// *const T -> *const U
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *const T {}


/// This is used for object safety, to check that a method's receiver type can be dispatched on.
///
/// An example implementation of the trait:
///
/// ```
/// # #![feature(dispatch_from_dyn, unsize)]
/// # use std::{ops::DispatchFromDyn, marker::Unsize};
/// # struct Rc<T: ?Sized>(::std::rc::Rc<T>);
/// impl<T: ?Sized, U: ?Sized> DispatchFromDyn<Rc<U>> for Rc<T>
/// where
///     T: Unsize<U>,
/// {}
/// ```
#[unstable(feature = "dispatch_from_dyn", issue = "0")]
#[lang = "dispatch_from_dyn"]
pub trait DispatchFromDyn<T> {
    // Empty.
}

// &T -> &U
#[unstable(feature = "dispatch_from_dyn", issue = "0")]
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> DispatchFromDyn<&'a U> for &'a T {}
// &mut T -> &mut U
#[unstable(feature = "dispatch_from_dyn", issue = "0")]
impl<'a, T: ?Sized+Unsize<U>, U: ?Sized> DispatchFromDyn<&'a mut U> for &'a mut T {}
// *const T -> *const U
#[unstable(feature = "dispatch_from_dyn", issue = "0")]
impl<T: ?Sized+Unsize<U>, U: ?Sized> DispatchFromDyn<*const U> for *const T {}
// *mut T -> *mut U
#[unstable(feature = "dispatch_from_dyn", issue = "0")]
impl<T: ?Sized+Unsize<U>, U: ?Sized> DispatchFromDyn<*mut U> for *mut T {}
