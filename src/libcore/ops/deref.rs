/// Used for immutable dereferencing operations, like `*v`.
///
/// In addition to being used for explicit dereferencing operations with the
/// (unary) `*` operator in immutable contexts, `Deref` is also used implicitly
/// by the compiler in many circumstances. This mechanism is called
/// ['`Deref` coercion'][more]. In mutable contexts, [`DerefMut`] is used.
///
/// Implementing `Deref` for smart pointers makes accessing the data behind them
/// convenient, which is why they implement `Deref`. On the other hand, the
/// rules regarding `Deref` and [`DerefMut`] were designed specifically to
/// accommodate smart pointers. Because of this, **`Deref` should only be
/// implemented for smart pointers** to avoid confusion.
///
/// For similar reasons, **this trait should never fail**. Failure during
/// dereferencing can be extremely confusing when `Deref` is invoked implicitly.
///
/// # More on `Deref` coercion
///
/// If `T` implements `Deref<Target = U>`, and `x` is a value of type `T`, then:
///
/// * In immutable contexts, `*x` on non-pointer types is equivalent to
///   `*Deref::deref(&x)`.
/// * Values of type `&T` are coerced to values of type `&U`
/// * `T` implicitly implements all the (immutable) methods of the type `U`.
///
/// For more details, visit [the chapter in *The Rust Programming Language*][book]
/// as well as the reference sections on [the dereference operator][ref-deref-op],
/// [method resolution] and [type coercions].
///
/// [book]: ../../book/ch15-02-deref.html
/// [`DerefMut`]: trait.DerefMut.html
/// [more]: #more-on-deref-coercion
/// [ref-deref-op]: ../../reference/expressions/operator-expr.html#the-dereference-operator
/// [method resolution]: ../../reference/expressions/method-call-expr.html
/// [type coercions]: ../../reference/type-coercions.html
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
pub trait Deref {
    /// The resulting type after dereferencing.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Target: ?Sized;

    /// Dereferences the value.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn deref(&self) -> &Self::Target;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Deref for &T {
    type Target = T;

    fn deref(&self) -> &T { *self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Deref for &mut T {
    type Target = T;

    fn deref(&self) -> &T { *self }
}

/// Used for mutable dereferencing operations, like in `*v = 1;`.
///
/// In addition to being used for explicit dereferencing operations with the
/// (unary) `*` operator in mutable contexts, `DerefMut` is also used implicitly
/// by the compiler in many circumstances. This mechanism is called
/// ['`Deref` coercion'][more]. In immutable contexts, [`Deref`] is used.
///
/// Implementing `DerefMut` for smart pointers makes mutating the data behind
/// them convenient, which is why they implement `DerefMut`. On the other hand,
/// the rules regarding [`Deref`] and `DerefMut` were designed specifically to
/// accommodate smart pointers. Because of this, **`DerefMut` should only be
/// implemented for smart pointers** to avoid confusion.
///
/// For similar reasons, **this trait should never fail**. Failure during
/// dereferencing can be extremely confusing when `DerefMut` is invoked
/// implicitly.
///
/// # More on `Deref` coercion
///
/// If `T` implements `DerefMut<Target = U>`, and `x` is a value of type `T`,
/// then:
///
/// * In mutable contexts, `*x` on non-pointer types is equivalent to
///   `*DerefMut::deref_mut(&mut x)`.
/// * Values of type `&mut T` are coerced to values of type `&mut U`
/// * `T` implicitly implements all the (mutable) methods of the type `U`.
///
/// For more details, visit [the chapter in *The Rust Programming Language*][book]
/// as well as the reference sections on [the dereference operator][ref-deref-op],
/// [method resolution] and [type coercions].
///
/// [book]: ../../book/ch15-02-deref.html
/// [`Deref`]: trait.Deref.html
/// [more]: #more-on-deref-coercion
/// [ref-deref-op]: ../../reference/expressions/operator-expr.html#the-dereference-operator
/// [method resolution]: ../../reference/expressions/method-call-expr.html
/// [type coercions]: ../../reference/type-coercions.html
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
/// assert_eq!('b', *x);
/// ```
#[lang = "deref_mut"]
#[doc(alias = "*")]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait DerefMut: Deref {
    /// Mutably dereferences the value.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn deref_mut(&mut self) -> &mut Self::Target;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> DerefMut for &mut T {
    fn deref_mut(&mut self) -> &mut T { *self }
}

/// Indicates that a struct can be used as a method receiver, without the
/// `arbitrary_self_types` feature. This is implemented by stdlib pointer types like `Box<T>`,
/// `Rc<T>`, `&T`, and `Pin<P>`.
#[lang = "receiver"]
#[unstable(feature = "receiver_trait", issue = "0")]
#[doc(hidden)]
pub trait Receiver {
    // Empty.
}

#[unstable(feature = "receiver_trait", issue = "0")]
impl<T: ?Sized> Receiver for &T {}

#[unstable(feature = "receiver_trait", issue = "0")]
impl<T: ?Sized> Receiver for &mut T {}
