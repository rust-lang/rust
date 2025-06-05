use crate::marker::PointeeSized;

/// Used for immutable dereferencing operations, like `*v`.
///
/// In addition to being used for explicit dereferencing operations with the
/// (unary) `*` operator in immutable contexts, `Deref` is also used implicitly
/// by the compiler in many circumstances. This mechanism is called
/// ["`Deref` coercion"][coercion]. In mutable contexts, [`DerefMut`] is used and
/// mutable deref coercion similarly occurs.
///
/// **Warning:** Deref coercion is a powerful language feature which has
/// far-reaching implications for every type that implements `Deref`. The
/// compiler will silently insert calls to `Deref::deref`. For this reason, one
/// should be careful about implementing `Deref` and only do so when deref
/// coercion is desirable. See [below][implementing] for advice on when this is
/// typically desirable or undesirable.
///
/// Types that implement `Deref` or `DerefMut` are often called "smart
/// pointers" and the mechanism of deref coercion has been specifically designed
/// to facilitate the pointer-like behavior that name suggests. Often, the
/// purpose of a "smart pointer" type is to change the ownership semantics
/// of a contained value (for example, [`Rc`][rc] or [`Cow`][cow]) or the
/// storage semantics of a contained value (for example, [`Box`][box]).
///
/// # Deref coercion
///
/// If `T` implements `Deref<Target = U>`, and `v` is a value of type `T`, then:
///
/// * In immutable contexts, `*v` (where `T` is neither a reference nor a raw
///   pointer) is equivalent to `*Deref::deref(&v)`.
/// * Values of type `&T` are coerced to values of type `&U`
/// * `T` implicitly implements all the methods of the type `U` which take the
///   `&self` receiver.
///
/// For more details, visit [the chapter in *The Rust Programming Language*][book]
/// as well as the reference sections on [the dereference operator][ref-deref-op],
/// [method resolution], and [type coercions].
///
/// # When to implement `Deref` or `DerefMut`
///
/// The same advice applies to both deref traits. In general, deref traits
/// **should** be implemented if:
///
/// 1. a value of the type transparently behaves like a value of the target
///    type;
/// 1. the implementation of the deref function is cheap; and
/// 1. users of the type will not be surprised by any deref coercion behavior.
///
/// In general, deref traits **should not** be implemented if:
///
/// 1. the deref implementations could fail unexpectedly; or
/// 1. the type has methods that are likely to collide with methods on the
///    target type; or
/// 1. committing to deref coercion as part of the public API is not desirable.
///
/// Note that there's a large difference between implementing deref traits
/// generically over many target types, and doing so only for specific target
/// types.
///
/// Generic implementations, such as for [`Box<T>`][box] (which is generic over
/// every type and dereferences to `T`) should be careful to provide few or no
/// methods, since the target type is unknown and therefore every method could
/// collide with one on the target type, causing confusion for users.
/// `impl<T> Box<T>` has no methods (though several associated functions),
/// partly for this reason.
///
/// Specific implementations, such as for [`String`][string] (whose `Deref`
/// implementation has `Target = str`) can have many methods, since avoiding
/// collision is much easier. `String` and `str` both have many methods, and
/// `String` additionally behaves as if it has every method of `str` because of
/// deref coercion. The implementing type may also be generic while the
/// implementation is still specific in this sense; for example, [`Vec<T>`][vec]
/// dereferences to `[T]`, so methods of `T` are not applicable.
///
/// Consider also that deref coercion means that deref traits are a much larger
/// part of a type's public API than any other trait as it is implicitly called
/// by the compiler. Therefore, it is advisable to consider whether this is
/// something you are comfortable supporting as a public API.
///
/// The [`AsRef`] and [`Borrow`][core::borrow::Borrow] traits have very similar
/// signatures to `Deref`. It may be desirable to implement either or both of
/// these, whether in addition to or rather than deref traits. See their
/// documentation for details.
///
/// # Fallibility
///
/// **This trait's method should never unexpectedly fail**. Deref coercion means
/// the compiler will often insert calls to `Deref::deref` implicitly. Failure
/// during dereferencing can be extremely confusing when `Deref` is invoked
/// implicitly. In the majority of uses it should be infallible, though it may
/// be acceptable to panic if the type is misused through programmer error, for
/// example.
///
/// However, infallibility is not enforced and therefore not guaranteed.
/// As such, `unsafe` code should not rely on infallibility in general for
/// soundness.
///
/// [book]: ../../book/ch15-02-deref.html
/// [coercion]: #deref-coercion
/// [implementing]: #when-to-implement-deref-or-derefmut
/// [ref-deref-op]: ../../reference/expressions/operator-expr.html#the-dereference-operator
/// [method resolution]: ../../reference/expressions/method-call-expr.html
/// [type coercions]: ../../reference/type-coercions.html
/// [box]: ../../alloc/boxed/struct.Box.html
/// [string]: ../../alloc/string/struct.String.html
/// [vec]: ../../alloc/vec/struct.Vec.html
/// [rc]: ../../alloc/rc/struct.Rc.html
/// [cow]: ../../alloc/borrow/enum.Cow.html
///
/// # Examples
///
/// A struct with a single field which is accessible by dereferencing the
/// struct.
///
/// ```
/// use std::ops::Deref;
///
/// struct DerefExample<T> {
///     value: T
/// }
///
/// impl<T> Deref for DerefExample<T> {
///     type Target = T;
///
///     fn deref(&self) -> &Self::Target {
///         &self.value
///     }
/// }
///
/// let x = DerefExample { value: 'a' };
/// assert_eq!('a', *x);
/// ```
#[lang = "deref"]
#[doc(alias = "*")]
#[doc(alias = "&*")]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "Deref"]
#[const_trait]
#[rustc_const_unstable(feature = "const_deref", issue = "88955")]
pub trait Deref: PointeeSized {
    /// The resulting type after dereferencing.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "deref_target"]
    #[lang = "deref_target"]
    type Target: ?Sized;

    /// Dereferences the value.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "deref_method"]
    fn deref(&self) -> &Self::Target;
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_deref", issue = "88955")]
impl<T: ?Sized> const Deref for &T {
    type Target = T;

    #[rustc_diagnostic_item = "noop_method_deref"]
    fn deref(&self) -> &T {
        *self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> !DerefMut for &T {}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_deref", issue = "88955")]
impl<T: ?Sized> const Deref for &mut T {
    type Target = T;

    fn deref(&self) -> &T {
        *self
    }
}

/// Used for mutable dereferencing operations, like in `*v = 1;`.
///
/// In addition to being used for explicit dereferencing operations with the
/// (unary) `*` operator in mutable contexts, `DerefMut` is also used implicitly
/// by the compiler in many circumstances. This mechanism is called
/// ["mutable deref coercion"][coercion]. In immutable contexts, [`Deref`] is used.
///
/// **Warning:** Deref coercion is a powerful language feature which has
/// far-reaching implications for every type that implements `DerefMut`. The
/// compiler will silently insert calls to `DerefMut::deref_mut`. For this
/// reason, one should be careful about implementing `DerefMut` and only do so
/// when mutable deref coercion is desirable. See [the `Deref` docs][implementing]
/// for advice on when this is typically desirable or undesirable.
///
/// Types that implement `DerefMut` or `Deref` are often called "smart
/// pointers" and the mechanism of deref coercion has been specifically designed
/// to facilitate the pointer-like behavior that name suggests. Often, the
/// purpose of a "smart pointer" type is to change the ownership semantics
/// of a contained value (for example, [`Rc`][rc] or [`Cow`][cow]) or the
/// storage semantics of a contained value (for example, [`Box`][box]).
///
/// # Mutable deref coercion
///
/// If `T` implements `DerefMut<Target = U>`, and `v` is a value of type `T`,
/// then:
///
/// * In mutable contexts, `*v` (where `T` is neither a reference nor a raw pointer)
///   is equivalent to `*DerefMut::deref_mut(&mut v)`.
/// * Values of type `&mut T` are coerced to values of type `&mut U`
/// * `T` implicitly implements all the (mutable) methods of the type `U`.
///
/// For more details, visit [the chapter in *The Rust Programming Language*][book]
/// as well as the reference sections on [the dereference operator][ref-deref-op],
/// [method resolution] and [type coercions].
///
/// # Fallibility
///
/// **This trait's method should never unexpectedly fail**. Deref coercion means
/// the compiler will often insert calls to `DerefMut::deref_mut` implicitly.
/// Failure during dereferencing can be extremely confusing when `DerefMut` is
/// invoked implicitly. In the majority of uses it should be infallible, though
/// it may be acceptable to panic if the type is misused through programmer
/// error, for example.
///
/// However, infallibility is not enforced and therefore not guaranteed.
/// As such, `unsafe` code should not rely on infallibility in general for
/// soundness.
///
/// [book]: ../../book/ch15-02-deref.html
/// [coercion]: #mutable-deref-coercion
/// [implementing]: Deref#when-to-implement-deref-or-derefmut
/// [ref-deref-op]: ../../reference/expressions/operator-expr.html#the-dereference-operator
/// [method resolution]: ../../reference/expressions/method-call-expr.html
/// [type coercions]: ../../reference/type-coercions.html
/// [box]: ../../alloc/boxed/struct.Box.html
/// [string]: ../../alloc/string/struct.String.html
/// [rc]: ../../alloc/rc/struct.Rc.html
/// [cow]: ../../alloc/borrow/enum.Cow.html
///
/// # Examples
///
/// A struct with a single field which is modifiable by dereferencing the
/// struct.
///
/// ```
/// use std::ops::{Deref, DerefMut};
///
/// struct DerefMutExample<T> {
///     value: T
/// }
///
/// impl<T> Deref for DerefMutExample<T> {
///     type Target = T;
///
///     fn deref(&self) -> &Self::Target {
///         &self.value
///     }
/// }
///
/// impl<T> DerefMut for DerefMutExample<T> {
///     fn deref_mut(&mut self) -> &mut Self::Target {
///         &mut self.value
///     }
/// }
///
/// let mut x = DerefMutExample { value: 'a' };
/// *x = 'b';
/// assert_eq!('b', x.value);
/// ```
#[lang = "deref_mut"]
#[doc(alias = "*")]
#[stable(feature = "rust1", since = "1.0.0")]
#[const_trait]
#[rustc_const_unstable(feature = "const_deref", issue = "88955")]
pub trait DerefMut: ~const Deref + PointeeSized {
    /// Mutably dereferences the value.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "deref_mut_method"]
    fn deref_mut(&mut self) -> &mut Self::Target;
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_deref", issue = "88955")]
impl<T: ?Sized> const DerefMut for &mut T {
    fn deref_mut(&mut self) -> &mut T {
        *self
    }
}

/// Perma-unstable marker trait. Indicates that the type has a well-behaved [`Deref`]
/// (and, if applicable, [`DerefMut`]) implementation. This is relied on for soundness
/// of deref patterns.
///
/// FIXME(deref_patterns): The precise semantics are undecided; the rough idea is that
/// successive calls to `deref`/`deref_mut` without intermediate mutation should be
/// idempotent, in the sense that they return the same value as far as pattern-matching
/// is concerned. Calls to `deref`/`deref_mut` must leave the pointer itself likewise
/// unchanged.
#[unstable(feature = "deref_pure_trait", issue = "87121")]
#[lang = "deref_pure"]
pub unsafe trait DerefPure: PointeeSized {}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl<T: ?Sized> DerefPure for &T {}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl<T: ?Sized> DerefPure for &mut T {}

/// Indicates that a struct can be used as a method receiver.
/// That is, a type can use this type as a type of `self`, like this:
/// ```compile_fail
/// # // This is currently compile_fail because the compiler-side parts
/// # // of arbitrary_self_types are not implemented
/// use std::ops::Receiver;
///
/// struct SmartPointer<T>(T);
///
/// impl<T> Receiver for SmartPointer<T> {
///    type Target = T;
/// }
///
/// struct MyContainedType;
///
/// impl MyContainedType {
///   fn method(self: SmartPointer<Self>) {
///     // ...
///   }
/// }
///
/// fn main() {
///   let ptr = SmartPointer(MyContainedType);
///   ptr.method();
/// }
/// ```
/// This trait is blanket implemented for any type which implements
/// [`Deref`], which includes stdlib pointer types like `Box<T>`,`Rc<T>`, `&T`,
/// and `Pin<P>`. For that reason, it's relatively rare to need to
/// implement this directly. You'll typically do this only if you need
/// to implement a smart pointer type which can't implement [`Deref`]; perhaps
/// because you're interfacing with another programming language and can't
/// guarantee that references comply with Rust's aliasing rules.
///
/// When looking for method candidates, Rust will explore a chain of possible
/// `Receiver`s, so for example each of the following methods work:
/// ```
/// use std::boxed::Box;
/// use std::rc::Rc;
///
/// // Both `Box` and `Rc` (indirectly) implement Receiver
///
/// struct MyContainedType;
///
/// fn main() {
///   let t = Rc::new(Box::new(MyContainedType));
///   t.method_a();
///   t.method_b();
///   t.method_c();
/// }
///
/// impl MyContainedType {
///   fn method_a(&self) {
///
///   }
///   fn method_b(self: &Box<Self>) {
///
///   }
///   fn method_c(self: &Rc<Box<Self>>) {
///
///   }
/// }
/// ```
#[lang = "receiver"]
#[unstable(feature = "arbitrary_self_types", issue = "44874")]
pub trait Receiver: PointeeSized {
    /// The target type on which the method may be called.
    #[rustc_diagnostic_item = "receiver_target"]
    #[lang = "receiver_target"]
    #[unstable(feature = "arbitrary_self_types", issue = "44874")]
    type Target: ?Sized;
}

#[unstable(feature = "arbitrary_self_types", issue = "44874")]
impl<P: ?Sized, T: ?Sized> Receiver for P
where
    P: Deref<Target = T>,
{
    type Target = T;
}

/// Indicates that a struct can be used as a method receiver, without the
/// `arbitrary_self_types` feature. This is implemented by stdlib pointer types like `Box<T>`,
/// `Rc<T>`, `&T`, and `Pin<P>`.
///
/// This trait will shortly be removed and replaced with a more generic
/// facility based around the current "arbitrary self types" unstable feature.
/// That new facility will use the replacement trait above called `Receiver`
/// which is why this is now named `LegacyReceiver`.
#[lang = "legacy_receiver"]
#[unstable(feature = "legacy_receiver_trait", issue = "none")]
#[doc(hidden)]
pub trait LegacyReceiver: PointeeSized {
    // Empty.
}

#[unstable(feature = "legacy_receiver_trait", issue = "none")]
impl<T: PointeeSized> LegacyReceiver for &T {}

#[unstable(feature = "legacy_receiver_trait", issue = "none")]
impl<T: PointeeSized> LegacyReceiver for &mut T {}
