//! Types which pin data to its location in memory
//!
//! It is sometimes useful to have objects that are guaranteed to not move,
//! in the sense that their placement in memory does not change, and can thus be relied upon.
//!
//! A prime example of such a scenario would be building self-referential structs,
//! since moving an object with pointers to itself will invalidate them,
//! which could cause undefined behavior.
//!
//! By default, all types in Rust are movable. Rust allows passing all types by-value,
//! and common smart-pointer types such as `Box`, `Rc`, and `&mut` allow replacing and
//! moving the values they contain. In order to prevent objects from moving, they must
//! be pinned by wrapping a pointer to the data in the [`Pin`] type.
//! Doing this prohibits moving the value behind the pointer.
//! For example, `Pin<Box<T>>` functions much like a regular `Box<T>`,
//! but doesn't allow moving `T`. The pointer value itself (the `Box`) can still be moved,
//! but the value behind it cannot.
//!
//! Since data can be moved out of `&mut` and `Box` with functions such as [`swap`],
//! changing the location of the underlying data, [`Pin`] prohibits accessing the
//! underlying pointer type (the `&mut` or `Box`) directly, and provides its own set of
//! APIs for accessing and using the value. [`Pin`] also guarantees that no other
//! functions will move the pointed-to value. This allows for the creation of
//! self-references and other special behaviors that are only possible for unmovable
//! values.
//!
//! However, these restrictions are usually not necessary. Many types are always freely
//! movable. These types implement the [`Unpin`] auto-trait, which nullifies the effect
//! of [`Pin`]. For `T: Unpin`, `Pin<Box<T>>` and `Box<T>` function identically, as do
//! `Pin<&mut T>` and `&mut T`.
//!
//! Note that pinning and `Unpin` only affect the pointed-to type. For example, whether
//! or not `Box<T>` is `Unpin` has no affect on the behavior of `Pin<Box<T>>`. Similarly,
//! `Pin<Box<T>>` and `Pin<&mut T>` are always `Unpin` themselves, even though the
//! `T` underneath them isn't, because the pointers in `Pin<Box<_>>` and `Pin<&mut _>`
//! are always freely movable, even if the data they point to isn't.
//!
//! [`Pin`]: struct.Pin.html
//! [`Unpin`]: ../../std/marker/trait.Unpin.html
//! [`swap`]: ../../std/mem/fn.swap.html
//! [`Box`]: ../../std/boxed/struct.Box.html
//!
//! # Examples
//!
//! ```rust
//! use std::pin::Pin;
//! use std::marker::PhantomPinned;
//! use std::ptr::NonNull;
//!
//! // This is a self-referential struct since the slice field points to the data field.
//! // We cannot inform the compiler about that with a normal reference,
//! // since this pattern cannot be described with the usual borrowing rules.
//! // Instead we use a raw pointer, though one which is known to not be null,
//! // since we know it's pointing at the string.
//! struct Unmovable {
//!     data: String,
//!     slice: NonNull<String>,
//!     _pin: PhantomPinned,
//! }
//!
//! impl Unmovable {
//!     // To ensure the data doesn't move when the function returns,
//!     // we place it in the heap where it will stay for the lifetime of the object,
//!     // and the only way to access it would be through a pointer to it.
//!     fn new(data: String) -> Pin<Box<Self>> {
//!         let res = Unmovable {
//!             data,
//!             // we only create the pointer once the data is in place
//!             // otherwise it will have already moved before we even started
//!             slice: NonNull::dangling(),
//!             _pin: PhantomPinned,
//!         };
//!         let mut boxed = Box::pin(res);
//!
//!         let slice = NonNull::from(&boxed.data);
//!         // we know this is safe because modifying a field doesn't move the whole struct
//!         unsafe {
//!             let mut_ref: Pin<&mut Self> = Pin::as_mut(&mut boxed);
//!             Pin::get_unchecked_mut(mut_ref).slice = slice;
//!         }
//!         boxed
//!     }
//! }
//!
//! let unmoved = Unmovable::new("hello".to_string());
//! // The pointer should point to the correct location,
//! // so long as the struct hasn't moved.
//! // Meanwhile, we are free to move the pointer around.
//! # #[allow(unused_mut)]
//! let mut still_unmoved = unmoved;
//! assert_eq!(still_unmoved.slice, NonNull::from(&still_unmoved.data));
//!
//! // Since our type doesn't implement Unpin, this will fail to compile:
//! // let new_unmoved = Unmovable::new("world".to_string());
//! // std::mem::swap(&mut *still_unmoved, &mut *new_unmoved);
//! ```

#![stable(feature = "pin", since = "1.33.0")]

use fmt;
use marker::{Sized, Unpin};
use cmp::{self, PartialEq, PartialOrd};
use ops::{Deref, DerefMut, Receiver, CoerceUnsized, DispatchFromDyn};

/// A pinned pointer.
///
/// This is a wrapper around a kind of pointer which makes that pointer "pin" its
/// value in place, preventing the value referenced by that pointer from being moved
/// unless it implements [`Unpin`].
///
/// See the [`pin` module] documentation for further explanation on pinning.
///
/// [`Unpin`]: ../../std/marker/trait.Unpin.html
/// [`pin` module]: ../../std/pin/index.html
//
// Note: the derives below, and the explicit `PartialEq` and `PartialOrd`
// implementations, are allowed because they all only use `&P`, so they cannot move
// the value behind `pointer`.
#[stable(feature = "pin", since = "1.33.0")]
#[cfg_attr(not(stage0), lang = "pin")]
#[fundamental]
#[repr(transparent)]
#[derive(Copy, Clone, Hash, Eq, Ord)]
pub struct Pin<P> {
    pointer: P,
}

#[stable(feature = "pin_partialeq_partialord_impl_applicability", since = "1.34.0")]
impl<P, Q> PartialEq<Pin<Q>> for Pin<P>
where
    P: PartialEq<Q>,
{
    fn eq(&self, other: &Pin<Q>) -> bool {
        self.pointer == other.pointer
    }

    fn ne(&self, other: &Pin<Q>) -> bool {
        self.pointer != other.pointer
    }
}

#[stable(feature = "pin_partialeq_partialord_impl_applicability", since = "1.34.0")]
impl<P, Q> PartialOrd<Pin<Q>> for Pin<P>
where
    P: PartialOrd<Q>,
{
    fn partial_cmp(&self, other: &Pin<Q>) -> Option<cmp::Ordering> {
        self.pointer.partial_cmp(&other.pointer)
    }

    fn lt(&self, other: &Pin<Q>) -> bool {
        self.pointer < other.pointer
    }

    fn le(&self, other: &Pin<Q>) -> bool {
        self.pointer <= other.pointer
    }

    fn gt(&self, other: &Pin<Q>) -> bool {
        self.pointer > other.pointer
    }

    fn ge(&self, other: &Pin<Q>) -> bool {
        self.pointer >= other.pointer
    }
}

impl<P: Deref> Pin<P>
where
    P::Target: Unpin,
{
    /// Construct a new `Pin` around a pointer to some data of a type that
    /// implements `Unpin`.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn new(pointer: P) -> Pin<P> {
        // Safety: the value pointed to is `Unpin`, and so has no requirements
        // around pinning.
        unsafe { Pin::new_unchecked(pointer) }
    }
}

impl<P: Deref> Pin<P> {
    /// Construct a new `Pin` around a reference to some data of a type that
    /// may or may not implement `Unpin`.
    ///
    /// # Safety
    ///
    /// This constructor is unsafe because we cannot guarantee that the data
    /// pointed to by `pointer` is pinned. If the constructed `Pin<P>` does
    /// not guarantee that the data `P` points to is pinned, constructing a
    /// `Pin<P>` is undefined behavior.
    ///
    /// If `pointer` dereferences to an `Unpin` type, `Pin::new` should be used
    /// instead.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub unsafe fn new_unchecked(pointer: P) -> Pin<P> {
        Pin { pointer }
    }

    /// Get a pinned shared reference from this pinned pointer.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn as_ref(self: &Pin<P>) -> Pin<&P::Target> {
        unsafe { Pin::new_unchecked(&*self.pointer) }
    }
}

impl<P: DerefMut> Pin<P> {
    /// Get a pinned mutable reference from this pinned pointer.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn as_mut(self: &mut Pin<P>) -> Pin<&mut P::Target> {
        unsafe { Pin::new_unchecked(&mut *self.pointer) }
    }

    /// Assign a new value to the memory behind the pinned reference.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn set(self: &mut Pin<P>, value: P::Target)
    where
        P::Target: Sized,
    {
        *(self.pointer) = value;
    }
}

impl<'a, T: ?Sized> Pin<&'a T> {
    /// Construct a new pin by mapping the interior value.
    ///
    /// For example, if you  wanted to get a `Pin` of a field of something,
    /// you could use this to get access to that field in one line of code.
    ///
    /// # Safety
    ///
    /// This function is unsafe. You must guarantee that the data you return
    /// will not move so long as the argument value does not move (for example,
    /// because it is one of the fields of that value), and also that you do
    /// not move out of the argument you receive to the interior function.
    #[stable(feature = "pin", since = "1.33.0")]
    pub unsafe fn map_unchecked<U, F>(self: Pin<&'a T>, func: F) -> Pin<&'a U> where
        F: FnOnce(&T) -> &U,
    {
        let pointer = &*self.pointer;
        let new_pointer = func(pointer);
        Pin::new_unchecked(new_pointer)
    }

    /// Get a shared reference out of a pin.
    ///
    /// Note: `Pin` also implements `Deref` to the target, which can be used
    /// to access the inner value. However, `Deref` only provides a reference
    /// that lives for as long as the borrow of the `Pin`, not the lifetime of
    /// the `Pin` itself. This method allows turning the `Pin` into a reference
    /// with the same lifetime as the original `Pin`.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn get_ref(self: Pin<&'a T>) -> &'a T {
        self.pointer
    }
}

impl<'a, T: ?Sized> Pin<&'a mut T> {
    /// Convert this `Pin<&mut T>` into a `Pin<&T>` with the same lifetime.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn into_ref(self: Pin<&'a mut T>) -> Pin<&'a T> {
        Pin { pointer: self.pointer }
    }

    /// Get a mutable reference to the data inside of this `Pin`.
    ///
    /// This requires that the data inside this `Pin` is `Unpin`.
    ///
    /// Note: `Pin` also implements `DerefMut` to the data, which can be used
    /// to access the inner value. However, `DerefMut` only provides a reference
    /// that lives for as long as the borrow of the `Pin`, not the lifetime of
    /// the `Pin` itself. This method allows turning the `Pin` into a reference
    /// with the same lifetime as the original `Pin`.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn get_mut(self: Pin<&'a mut T>) -> &'a mut T
        where T: Unpin,
    {
        self.pointer
    }

    /// Get a mutable reference to the data inside of this `Pin`.
    ///
    /// # Safety
    ///
    /// This function is unsafe. You must guarantee that you will never move
    /// the data out of the mutable reference you receive when you call this
    /// function, so that the invariants on the `Pin` type can be upheld.
    ///
    /// If the underlying data is `Unpin`, `Pin::get_mut` should be used
    /// instead.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(self: Pin<&'a mut T>) -> &'a mut T {
        self.pointer
    }

    /// Construct a new pin by mapping the interior value.
    ///
    /// For example, if you  wanted to get a `Pin` of a field of something,
    /// you could use this to get access to that field in one line of code.
    ///
    /// # Safety
    ///
    /// This function is unsafe. You must guarantee that the data you return
    /// will not move so long as the argument value does not move (for example,
    /// because it is one of the fields of that value), and also that you do
    /// not move out of the argument you receive to the interior function.
    #[stable(feature = "pin", since = "1.33.0")]
    pub unsafe fn map_unchecked_mut<U, F>(self: Pin<&'a mut T>, func: F) -> Pin<&'a mut U> where
        F: FnOnce(&mut T) -> &mut U,
    {
        let pointer = Pin::get_unchecked_mut(self);
        let new_pointer = func(pointer);
        Pin::new_unchecked(new_pointer)
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<P: Deref> Deref for Pin<P> {
    type Target = P::Target;
    fn deref(&self) -> &P::Target {
        Pin::get_ref(Pin::as_ref(self))
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<P: DerefMut> DerefMut for Pin<P>
where
    P::Target: Unpin
{
    fn deref_mut(&mut self) -> &mut P::Target {
        Pin::get_mut(Pin::as_mut(self))
    }
}

#[unstable(feature = "receiver_trait", issue = "0")]
impl<P: Receiver> Receiver for Pin<P> {}

#[stable(feature = "pin", since = "1.33.0")]
impl<P: fmt::Debug> fmt::Debug for Pin<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.pointer, f)
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<P: fmt::Display> fmt::Display for Pin<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.pointer, f)
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<P: fmt::Pointer> fmt::Pointer for Pin<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.pointer, f)
    }
}

// Note: this means that any impl of `CoerceUnsized` that allows coercing from
// a type that impls `Deref<Target=impl !Unpin>` to a type that impls
// `Deref<Target=Unpin>` is unsound. Any such impl would probably be unsound
// for other reasons, though, so we just need to take care not to allow such
// impls to land in std.
#[stable(feature = "pin", since = "1.33.0")]
impl<P, U> CoerceUnsized<Pin<U>> for Pin<P>
where
    P: CoerceUnsized<U>,
{}

#[stable(feature = "pin", since = "1.33.0")]
impl<'a, P, U> DispatchFromDyn<Pin<U>> for Pin<P>
where
    P: DispatchFromDyn<U>,
{}
