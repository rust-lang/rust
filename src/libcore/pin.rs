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
//! Since data can be moved out of `&mut` and `Box` with functions such as [`mem::swap`],
//! changing the location of the underlying data, [`Pin`] prohibits accessing the
//! underlying pointer type (the `&mut` or `Box`) directly, and provides its own set of
//! APIs for accessing and using the value. [`Pin`] also guarantees that no other
//! functions will move the pointed-to value. This allows for the creation of
//! self-references and other special behaviors that are only possible for unmovable
//! values.
//!
//! It is worth reiterating that [`Pin`] does *not* change the fact that the Rust compiler
//! considers all types movable.  [`mem::swap`] remains callable for any `T`.  Instead, `Pin`
//! prevents certain *values* (pointed to by pointers wrapped in `Pin`) from being
//! moved by making it impossible to call methods like [`mem::swap`] on them.
//!
//! # `Unpin`
//!
//! However, these restrictions are usually not necessary. Many types are always freely
//! movable, even when pinned. These types implement the [`Unpin`] auto-trait, which
//! nullifies the effect of [`Pin`]. For `T: Unpin`, `Pin<Box<T>>` and `Box<T>` function
//! identically, as do `Pin<&mut T>` and `&mut T`.
//!
//! Note that pinning and `Unpin` only affect the pointed-to type, not the pointer
//! type itself that got wrapped in `Pin`. For example, whether or not `Box<T>` is
//! `Unpin` has no affect on the behavior of `Pin<Box<T>>` (here, `T` is the
//! pointed-to type).
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
//!
//! # `Drop` guarantee
//!
//! The purpose of pinning is to be able to rely on the placement of some data in memory.
//! To make this work, not just moving the data is restricted; deallocating or overwriting
//! it is restricted, too. Concretely, for pinned data you have to maintain the invariant
//! that *it will not get overwritten or deallocated until `drop` was called*.
//! ("Overwriting" here refers to other ways of invalidating storage, such as switching
//! from one enum variant to another.)
//!
//! The purpose of this guarantee is to allow data structures that store pointers
//! to pinned data. For example, in an intrusive doubly-linked list, every element
//! will have pointers to its predecessor and successor in the list. Every element
//! will be pinned, because moving the elements around would invalidate the pointers.
//! Moreover, the `Drop` implemenetation of a linked list element will patch the pointers
//! of its predecessor and successor to remove itself from the list. Clearly, if an element
//! could be deallocated or overwritten without calling `drop`, the pointers into it
//! from its neighbouring elements would become invalid, breaking the data structure.
//!
//! Notice that this guarantee does *not* mean that memory does not leak! It is still
//! completely okay not to ever call `drop` on a pinned element (e.g., you can still
//! call [`mem::forget`] on a `Pin<Box<T>>`). What you may not do is free or reuse the storage
//! without calling `drop`.
//!
//! # `Drop` implementation
//!
//! If your type relies on pinning (for example, because it contains internal
//! references, or because you are implementing something like the intrusive
//! doubly-linked list mentioned in the previous section), you have to be careful
//! when implementing `Drop`: notice that `drop` takes `&mut self`, but this
//! will be called even if your type was previously pinned! It is as if the
//! compiler automatically called `get_unchecked_mut`. This can never cause
//! a problem in safe code because implementing a type that relies on pinning
//! requires unsafe code, but be aware that deciding to make use of pinning
//! in your type (for example by implementing some operation on `Pin<&[mut] Self>`)
//! has consequences for your `Drop` implemenetation as well.
//!
//! # Projections and Structural Pinning
//!
//! One interesting question arises when considering pinning and "container types" --
//! types such as `Vec` or `Box` but also `RefCell`; types that serve as wrappers
//! around other types.  When can such a type have a "projection" operation, an
//! operation with type `fn(Pin<&[mut] Container<T>>) -> Pin<&[mut] T>`?
//! This does not just apply to generic container types, even for normal structs
//! the question arises whether `fn(Pin<&[mut] Struct>) -> Pin<&[mut] Field>`
//! is an operation that can be soundly added to the API.
//!
//! This question is closely related to the question of whether pinning is "structural":
//! when you have pinned a container, have you pinned its contents? Adding a
//! projection to the API answers that question with a "yes" by offering pinned access
//! to the contents.
//!
//! In general, as the author of a type you get to decide whether pinning is structural, and
//! whether projections are provided. However, there are a couple requirements to be
//! upheld when adding projection operations:
//!
//! 1. The container must only be [`Unpin`] if all its fields are `Unpin`. This is the default,
//!    but `Unpin` is a safe trait, so as the author of the container it is your responsibility
//!    *not* to add something like `impl<T> Unpin for Container<T>`. (Notice that adding a
//!    projection operation requires unsafe code, so the fact that `Unpin` is a safe trait
//!    does not break the principle that you only have to worry about any of this if
//!    you use `unsafe`.)
//! 2. The destructor of the container must not move out of its argument. This is the exact
//!    point that was raised in the [previous section][drop-impl]: `drop` takes `&mut self`,
//!    but the container (and hence its fields) might have been pinned before.
//!    You have to guarantee that you do not move a field inside your `Drop` implementation.
//! 3. Your container type must *not* be `#[repr(packed)]`. Packed structs have their fields
//!    moved around when they are dropped to properly align them, which is in conflict with
//!    claiming that the fields are pinned when your struct is.
//! 4. You must make sure that you uphold the [`Drop` guarantee][drop-guarantee]:
//!    you must make sure that, once your container is pinned, the memory containing the
//!    content is not overwritten or deallocated without calling the content's destructors.
//!    This can be tricky, as witnessed by `VecDeque`: the destructor of `VecDeque` can fail
//!    to call `drop` on all elements if one of the destructors panics. This violates the
//!    `Drop` guarantee, because it can lead to elements being deallocated without
//!    their destructor being called.
//! 5. You must not offer any other operations that could lead to data being moved out of
//!    the fields when your type is pinned. This is usually not a concern, but can become
//!    tricky when interior mutability is involved. For example, imagine `RefCell`
//!    would have a method `fn get_pin_mut(self: Pin<&mut Self>) -> Pin<&mut T>`.
//!    This would be catastrophic, because it is possible to move out of a pinned
//!    `RefCell`: from `x: Pin<&mut RefCell<T>>`, use `let y = x.into_ref().get_ref()` to obtain
//!    `y: &RefCell<T>`, and from there use `y.borrow_mut().deref_mut()` to obtain `&mut T`
//!    which can be used with [`mem::swap`].
//!
//! On the other hand, if you decide *not* to offer any pinning projections, you
//! are free to do `impl<T> Unpin for Container<T>`.  In the standard library,
//! we do this for all pointer types: `Box<T>: Unpin` holds for all `T`.
//! It makes a lot of sense to do this for pointer types, because moving the `Box<T>`
//! does not actually move the `T`: the `Box<T>` can be freely movable even if the `T`
//! is not. In fact, even `Pin<Box<T>>` and `Pin<&mut T>` are always `Unpin` themselves,
//! for the same reason.
//!
//! [`Pin`]: struct.Pin.html
//! [`Unpin`]: ../../std/marker/trait.Unpin.html
//! [`mem::swap`]: ../../std/mem/fn.swap.html
//! [`mem::forget`]: ../../std/mem/fn.forget.html
//! [`Box`]: ../../std/boxed/struct.Box.html
//! [drop-impl]: #drop-implementation
//! [drop-guarantee]: #drop-guarantee

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
    /// implements [`Unpin`].
    ///
    /// Unlike `Pin::new_unchecked`, this method is safe because the pointer
    /// `P` dereferences to an [`Unpin`] type, which nullifies the pinning guarantees.
    ///
    /// [`Unpin`]: ../../std/marker/trait.Unpin.html
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
    /// By using this method, you are making a promise about the `P::Deref` and
    /// `P::DerefMut` implementations, if they exist. Most importantly, they
    /// must not move out of their `self` arguments: `Pin::as_mut` and `Pin::as_ref`
    /// will call `DerefMut::deref_mut` and `Deref::deref` *on the pinned pointer*
    /// and expect these methods to uphold the pinning invariants.
    /// Moreover, by calling this method you promise that the reference `P`
    /// dereferences to will not be moved out of again; in particular, it
    /// must not be possible to obtain a `&mut P::Target` and then
    /// move out of that reference (using, for example [`replace`]).
    ///
    /// For example, the following is a *violation* of `Pin`'s safety:
    /// ```
    /// use std::mem;
    /// use std::pin::Pin;
    ///
    /// fn foo<T>(mut a: T, b: T) {
    ///     unsafe { let p = Pin::new_unchecked(&mut a); } // should mean `a` can never move again
    ///     let a2 = mem::replace(&mut a, b);
    ///     // the address of `a` changed to `a2`'s stack slot, so `a` got moved even
    ///     // though we have previously pinned it!
    /// }
    /// ```
    ///
    /// If `pointer` dereferences to an `Unpin` type, `Pin::new` should be used
    /// instead.
    ///
    /// [`replace`]: ../../std/mem/fn.replace.html
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub unsafe fn new_unchecked(pointer: P) -> Pin<P> {
        Pin { pointer }
    }

    /// Gets a pinned shared reference from this pinned pointer.
    ///
    /// This is a generic method to go from `&Pin<SmartPointer<T>>` to `Pin<&T>`.
    /// It is safe because, as part of the contract of `Pin::new_unchecked`,
    /// the pointee cannot move after `Pin<SmartPointer<T>>` got created.
    /// "Malicious" implementations of `SmartPointer::Deref` are likewise
    /// ruled out by the contract of `Pin::new_unchecked`.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn as_ref(self: &Pin<P>) -> Pin<&P::Target> {
        unsafe { Pin::new_unchecked(&*self.pointer) }
    }
}

impl<P: DerefMut> Pin<P> {
    /// Gets a pinned mutable reference from this pinned pointer.
    ///
    /// This is a generic method to go from `&mut Pin<SmartPointer<T>>` to `Pin<&mut T>`.
    /// It is safe because, as part of the contract of `Pin::new_unchecked`,
    /// the pointee cannot move after `Pin<SmartPointer<T>>` got created.
    /// "Malicious" implementations of `SmartPointer::DerefMut` are likewise
    /// ruled out by the contract of `Pin::new_unchecked`.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn as_mut(self: &mut Pin<P>) -> Pin<&mut P::Target> {
        unsafe { Pin::new_unchecked(&mut *self.pointer) }
    }

    /// Assigns a new value to the memory behind the pinned reference.
    ///
    /// This overwrites pinned data, but that is okay: its destructor gets
    /// run before being overwritten, so no pinning guarantee is violated.
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
    /// Constructs a new pin by mapping the interior value.
    ///
    /// For example, if you  wanted to get a `Pin` of a field of something,
    /// you could use this to get access to that field in one line of code.
    /// However, there are several gotchas with these "pinning projections";
    /// see the [`pin` module] documentation for further details on that topic.
    ///
    /// # Safety
    ///
    /// This function is unsafe. You must guarantee that the data you return
    /// will not move so long as the argument value does not move (for example,
    /// because it is one of the fields of that value), and also that you do
    /// not move out of the argument you receive to the interior function.
    ///
    /// [`pin` module]: ../../std/pin/index.html#projections-and-structural-pinning
    #[stable(feature = "pin", since = "1.33.0")]
    pub unsafe fn map_unchecked<U, F>(self: Pin<&'a T>, func: F) -> Pin<&'a U> where
        F: FnOnce(&T) -> &U,
    {
        let pointer = &*self.pointer;
        let new_pointer = func(pointer);
        Pin::new_unchecked(new_pointer)
    }

    /// Gets a shared reference out of a pin.
    ///
    /// This is safe because it is not possible to move out of a shared reference.
    /// It may seem like there is an issue here with interior mutability: in fact,
    /// it *is* possible to move a `T` out of a `&RefCell<T>`. However, this is
    /// not a problem as long as there does not also exist a `Pin<&T>` pointing
    /// to the same data, and `RefCell` does not let you create a pinned reference
    /// to its contents. See the discussion on ["pinning projections"] for further
    /// details.
    ///
    /// Note: `Pin` also implements `Deref` to the target, which can be used
    /// to access the inner value. However, `Deref` only provides a reference
    /// that lives for as long as the borrow of the `Pin`, not the lifetime of
    /// the `Pin` itself. This method allows turning the `Pin` into a reference
    /// with the same lifetime as the original `Pin`.
    ///
    /// ["pinning projections"]: ../../std/pin/index.html#projections-and-structural-pinning
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn get_ref(self: Pin<&'a T>) -> &'a T {
        self.pointer
    }
}

impl<'a, T: ?Sized> Pin<&'a mut T> {
    /// Converts this `Pin<&mut T>` into a `Pin<&T>` with the same lifetime.
    #[stable(feature = "pin", since = "1.33.0")]
    #[inline(always)]
    pub fn into_ref(self: Pin<&'a mut T>) -> Pin<&'a T> {
        Pin { pointer: self.pointer }
    }

    /// Gets a mutable reference to the data inside of this `Pin`.
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

    /// Gets a mutable reference to the data inside of this `Pin`.
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
    /// However, there are several gotchas with these "pinning projections";
    /// see the [`pin` module] documentation for further details on that topic.
    ///
    /// # Safety
    ///
    /// This function is unsafe. You must guarantee that the data you return
    /// will not move so long as the argument value does not move (for example,
    /// because it is one of the fields of that value), and also that you do
    /// not move out of the argument you receive to the interior function.
    ///
    /// [`pin` module]: ../../std/pin/index.html#projections-and-structural-pinning
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
