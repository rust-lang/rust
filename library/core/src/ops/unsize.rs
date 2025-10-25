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
/// Note: `DispatchFromDyn` was briefly named `CoerceSized` (and had a slightly different
/// interpretation).
///
/// Imagine we have a trait object `t` with type `&dyn Tr`, where `Tr` is some trait with a method
/// `m` defined as `fn m(&self);`. When calling `t.m()`, the receiver `t` is a wide pointer, but an
/// implementation of `m` will expect a narrow pointer as `&self` (a reference to the concrete
/// type). The compiler must generate an implicit conversion from the trait object/wide pointer to
/// the concrete reference/narrow pointer. Implementing `DispatchFromDyn` indicates that that
/// conversion is allowed and thus that the type implementing `DispatchFromDyn` is safe to use as
/// the self type in an dyn-compatible method. (in the above example, the compiler will require
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
///
/// [^1]: Formerly known as *object safety*.
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

/// Derive macro that makes a smart pointer usable with trait objects.
///
/// # What this macro does
///
/// This macro is intended to be used with user-defined pointer types, and makes it possible to
/// perform coercions on the pointee of the user-defined pointer. There are two aspects to this:
///
/// ## Unsizing coercions of the pointee
///
/// By using the macro, the following example will compile:
/// ```
/// #![feature(derive_coerce_pointee)]
/// use std::ops::CoercePointee;
/// use std::ops::Deref;
///
/// #[derive(CoercePointee)]
/// #[repr(transparent)]
/// struct MySmartPointer<T: ?Sized>(Box<T>);
///
/// impl<T: ?Sized> Deref for MySmartPointer<T> {
///     type Target = T;
///     fn deref(&self) -> &T {
///         &self.0
///     }
/// }
///
/// trait MyTrait {}
///
/// impl MyTrait for i32 {}
///
/// fn main() {
///     let ptr: MySmartPointer<i32> = MySmartPointer(Box::new(4));
///
///     // This coercion would be an error without the derive.
///     let ptr: MySmartPointer<dyn MyTrait> = ptr;
/// }
/// ```
/// Without the `#[derive(CoercePointee)]` macro, this example would fail with the following error:
/// ```text
/// error[E0308]: mismatched types
///   --> src/main.rs:11:44
///    |
/// 11 |     let ptr: MySmartPointer<dyn MyTrait> = ptr;
///    |              ---------------------------   ^^^ expected `MySmartPointer<dyn MyTrait>`, found `MySmartPointer<i32>`
///    |              |
///    |              expected due to this
///    |
///    = note: expected struct `MySmartPointer<dyn MyTrait>`
///               found struct `MySmartPointer<i32>`
///    = help: `i32` implements `MyTrait` so you could box the found value and coerce it to the trait object `Box<dyn MyTrait>`, you will have to change the expected type as well
/// ```
///
/// ## Dyn compatibility
///
/// This macro allows you to dispatch on the user-defined pointer type. That is, traits using the
/// type as a receiver are dyn-compatible. For example, this compiles:
///
/// ```
/// #![feature(arbitrary_self_types, derive_coerce_pointee)]
/// use std::ops::CoercePointee;
/// use std::ops::Deref;
///
/// #[derive(CoercePointee)]
/// #[repr(transparent)]
/// struct MySmartPointer<T: ?Sized>(Box<T>);
///
/// impl<T: ?Sized> Deref for MySmartPointer<T> {
///     type Target = T;
///     fn deref(&self) -> &T {
///         &self.0
///     }
/// }
///
/// // You can always define this trait. (as long as you have #![feature(arbitrary_self_types)])
/// trait MyTrait {
///     fn func(self: MySmartPointer<Self>);
/// }
///
/// // But using `dyn MyTrait` requires #[derive(CoercePointee)].
/// fn call_func(value: MySmartPointer<dyn MyTrait>) {
///     value.func();
/// }
/// ```
/// If you remove the `#[derive(CoercePointee)]` annotation from the struct, then the above example
/// will fail with this error message:
/// ```text
/// error[E0038]: the trait `MyTrait` is not dyn compatible
///   --> src/lib.rs:21:36
///    |
/// 17 |     fn func(self: MySmartPointer<Self>);
///    |                   -------------------- help: consider changing method `func`'s `self` parameter to be `&self`: `&Self`
/// ...
/// 21 | fn call_func(value: MySmartPointer<dyn MyTrait>) {
///    |                                    ^^^^^^^^^^^ `MyTrait` is not dyn compatible
///    |
/// note: for a trait to be dyn compatible it needs to allow building a vtable
///       for more information, visit <https://doc.rust-lang.org/reference/items/traits.html#object-safety>
///   --> src/lib.rs:17:19
///    |
/// 16 | trait MyTrait {
///    |       ------- this trait is not dyn compatible...
/// 17 |     fn func(self: MySmartPointer<Self>);
///    |                   ^^^^^^^^^^^^^^^^^^^^ ...because method `func`'s `self` parameter cannot be dispatched on
/// ```
///
/// # Requirements for using the macro
///
/// This macro can only be used if:
/// * The type is a `#[repr(transparent)]` struct.
/// * The type of its non-zero-sized field must either be a standard library pointer type
///   (reference, raw pointer, `NonNull`, `Box`, `Rc`, `Arc`, etc.) or another user-defined type
///   also using the `#[derive(CoercePointee)]` macro.
/// * Zero-sized fields must not mention any generic parameters unless the zero-sized field has
///   type [`PhantomData`](crate::marker::PhantomData).
///
/// ## Multiple type parameters
///
/// If the type has multiple type parameters, then you must explicitly specify which one should be
/// used for dynamic dispatch. For example:
/// ```
/// # #![feature(derive_coerce_pointee)]
/// # use std::marker::PhantomData;
/// # use std::ops::CoercePointee;
/// #[derive(CoercePointee)]
/// #[repr(transparent)]
/// struct MySmartPointer<#[pointee] T: ?Sized, U> {
///     ptr: Box<T>,
///     _phantom: PhantomData<U>,
/// }
/// ```
/// Specifying `#[pointee]` when the struct has only one type parameter is allowed, but not required.
///
/// # Examples
///
/// A custom implementation of the `Rc` type:
/// ```
/// #![feature(derive_coerce_pointee)]
/// use std::ops::CoercePointee;
/// use std::ops::Deref;
/// use std::ptr::NonNull;
///
/// #[derive(CoercePointee)]
/// #[repr(transparent)]
/// pub struct Rc<T: ?Sized> {
///     inner: NonNull<RcInner<T>>,
/// }
///
/// struct RcInner<T: ?Sized> {
///     refcount: usize,
///     value: T,
/// }
///
/// impl<T: ?Sized> Deref for Rc<T> {
///     type Target = T;
///     fn deref(&self) -> &T {
///         let ptr = self.inner.as_ptr();
///         unsafe { &(*ptr).value }
///     }
/// }
///
/// impl<T> Rc<T> {
///     pub fn new(value: T) -> Self {
///         let inner = Box::new(RcInner {
///             refcount: 1,
///             value,
///         });
///         Self {
///             inner: NonNull::from(Box::leak(inner)),
///         }
///     }
/// }
///
/// impl<T: ?Sized> Clone for Rc<T> {
///     fn clone(&self) -> Self {
///         // A real implementation would handle overflow here.
///         unsafe { (*self.inner.as_ptr()).refcount += 1 };
///         Self { inner: self.inner }
///     }
/// }
///
/// impl<T: ?Sized> Drop for Rc<T> {
///     fn drop(&mut self) {
///         let ptr = self.inner.as_ptr();
///         unsafe { (*ptr).refcount -= 1 };
///         if unsafe { (*ptr).refcount } == 0 {
///             drop(unsafe { Box::from_raw(ptr) });
///         }
///     }
/// }
/// ```
#[rustc_builtin_macro(CoercePointee, attributes(pointee))]
#[allow_internal_unstable(dispatch_from_dyn, coerce_unsized, unsize, coerce_pointee_validated)]
#[rustc_diagnostic_item = "CoercePointee"]
#[unstable(feature = "derive_coerce_pointee", issue = "123430")]
pub macro CoercePointee($item:item) {
    /* compiler built-in */
}

/// A trait that is implemented for ADTs with `derive(CoercePointee)` so that
/// the compiler can enforce the derive impls are valid post-expansion, since
/// the derive has stricter requirements than if the impls were written by hand.
///
/// This trait is not intended to be implemented by users or used other than
/// validation, so it should never be stabilized.
#[lang = "coerce_pointee_validated"]
#[unstable(feature = "coerce_pointee_validated", issue = "none")]
#[doc(hidden)]
pub trait CoercePointeeValidated {
    /* compiler built-in */
}
