//! Types which pin data to its location in memory
//!
//! It is sometimes useful to have objects that are guaranteed to not move,
//! in the sense that their placement in memory does not change, and can thus be relied upon.
//!
//! A prime example of such a scenario would be building self-referencial structs,
//! since moving an object with pointers to itself will invalidate them,
//! which could cause undefined behavior.
//!
//! In order to prevent objects from moving, they must be *pinned*,
//! by wrapping a pointer to the data in the [`Pin`] type. A pointer wrapped
//! in a `Pin` is otherwise equivalent to its normal version, e.g. `Pin<Box<T>>`
//! and `Box<T>` work the same way except that the first is pinning the value
//! of `T` in place.
//!
//! First of all, these are pointer types because pinned data mustn't be passed around by value
//! (that would change its location in memory).
//! Secondly, since data can be moved out of `&mut` and [`Box`] with functions such as [`swap`],
//! which causes their contents to swap places in memory,
//! we need dedicated types that prohibit such operations.
//!
//! However, these restrictions are usually not necessary,
//! so most types implement the [`Unpin`] auto-trait,
//! which indicates that the type can be moved out safely.
//! Doing so removes the limitations of pinning types,
//! making them the same as their non-pinning counterparts.
//!
//! [`Pin`]: struct.Pin.html
//! [`Unpin`]: trait.Unpin.html
//! [`swap`]: ../../std/mem/fn.swap.html
//! [`Box`]: ../boxed/struct.Box.html
//!
//! # Examples
//!
//! ```rust
//! #![feature(pin)]
//!
//! use std::pin::Pin;
//! use std::marker::Pinned;
//! use std::ptr::NonNull;
//!
//! // This is a self referencial struct since the slice field points to the data field.
//! // We cannot inform the compiler about that with a normal reference,
//! // since this pattern cannot be described with the usual borrowing rules.
//! // Instead we use a raw pointer, though one which is known to not be null,
//! // since we know it's pointing at the string.
//! struct Unmovable {
//!     data: String,
//!     slice: NonNull<String>,
//!     _pin: Pinned,
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
//!             _pin: Pinned,
//!         };
//!         let mut boxed = Box::pinned(res);
//!
//!         let slice = NonNull::from(&boxed.data);
//!         // we know this is safe because modifying a field doesn't move the whole struct
//!         unsafe { 
//!             let mut_ref: Pin<&mut Self> = Pin::as_mut(&mut boxed);
//!             Pin::get_mut_unchecked(mut_ref).slice = slice;
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

#![unstable(feature = "pin", issue = "49150")]

use fmt;
use marker::{Sized, Unpin, Unsize};
use ops::{Deref, DerefMut, CoerceUnsized};

/// A pinned pointer.
///
/// This is a wrapper around a kind of pointer which makes that pointer "pin" its
/// value in place, preventing the value referenced by that pointer from being moved
/// unless it implements [`Unpin`].
///
/// See the [`pin` module] documentation for furthur explanation on pinning.
///
/// [`Unpin`]: ../../std/marker/trait.Unpin.html
/// [`pin` module]: ../../std/pin/index.html
#[unstable(feature = "pin", issue = "49150")]
#[fundamental]
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Pin<P> {
    pointer: P,
}

impl<P, T> Pin<P> where
    P: Deref<Target = T>,
    T: ?Sized + Unpin,
{
    /// Construct a new `Pin` around a pointer to some data of a type that
    /// implements `Unpin`.
    #[unstable(feature = "pin", issue = "49150")]
    pub fn new(pointer: P) -> Pin<P> {
        unsafe { Pin::new_unchecked(pointer) }
    }
}

impl<P, T> Pin<P> where
    P: Deref<Target = T>,
    T: ?Sized,
{
    /// Construct a new `Pin` around a reference to some data of a type that
    /// may or may not implement `Unpin`.
    ///
    /// # Safety
    ///
    /// This constructor is unsafe because we cannot guarantee that the target data
    /// is properly pinned by this pointer. If the constructed `Pin<P>` does not guarantee
    /// that the data is "pinned," constructing a `Pin<P>` is undefined behavior and could lead
    /// to segmentation faults or worse.
    #[unstable(feature = "pin", issue = "49150")]
    pub unsafe fn new_unchecked(pointer: P) -> Pin<P> {
        Pin { pointer }
    }


    /// Get a pinned shared reference from this pinned pointer.
    #[unstable(feature = "pin", issue = "49150")]
    pub fn as_ref(this: &Pin<P>) -> Pin<&T> {
        unsafe { Pin::new_unchecked(&**this) }
    }
}

impl<P, T> Pin<P> where
    P: DerefMut<Target = T>,
    T: ?Sized,
{
    /// Get a pinned mutable reference from this pinned pointer.
    #[unstable(feature = "pin", issue = "49150")]
    pub fn as_mut(this: &mut Pin<P>) -> Pin<&mut T> {
        unsafe { Pin::new_unchecked(&mut *this.pointer) }
    }

    /// Assign a new value to the memory behind the pinned reference.
    #[unstable(feature = "pin", issue = "49150")]
    pub fn set(this: Pin<&mut T>, value: T)
        where T: Sized,
    {
        *this.pointer = value;
    }
}

impl<'a, T> Pin<&'a T> {
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
    #[unstable(feature = "pin", issue = "49150")]
    pub unsafe fn map_unchecked<U, F>(this: Pin<&'a T>, func: F) -> Pin<&'a U> where
        F: FnOnce(&T) -> &U,
    {
        let pointer = &*this.pointer;
        let new_pointer = func(pointer);
        Pin::new_unchecked(new_pointer)
    }

    /// Get a safe reference out of a pin.
    #[unstable(feature = "pin", issue = "49150")]
    pub fn get(this: Pin<&'a T>) -> &'a T {
        this.pointer
    }
}

impl<'a, T> Pin<&'a mut T> {
    /// Get a mutable reference to the data inside of this `Pin`.
    ///
    /// # Safety
    ///
    /// This function is unsafe. You must guarantee that you will never move
    /// the data out of the mutable reference you receive when you call this
    /// function, so that the invariants on the `Pin` type can be upheld.
    #[unstable(feature = "pin", issue = "49150")]
    pub unsafe fn get_mut_unchecked(this: Pin<&'a mut T>) -> &'a mut T {
        this.pointer
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
    #[unstable(feature = "pin", issue = "49150")]
    pub unsafe fn map_unchecked_mut<U, F>(this: Pin<&'a mut T>, func: F) -> Pin<&'a mut U> where
        F: FnOnce(&mut T) -> &mut U,
    {
        let pointer = Pin::get_mut_unchecked(this);
        let new_pointer = func(pointer);
        Pin::new_unchecked(new_pointer)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<P, T> Deref for Pin<P> where
    P: Deref<Target = T>,
    T: ?Sized,
{
    type Target = T;
    fn deref(&self) -> &T {
        &*self.pointer
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<P, T> DerefMut for Pin<P> where
    P: DerefMut<Target = T>,
    T: ?Sized + Unpin,
{
    fn deref_mut(&mut self) -> &mut T {
        &mut *self.pointer
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, P: fmt::Debug> fmt::Debug for Pin<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.pointer, f)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, P: fmt::Display> fmt::Display for Pin<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.pointer, f)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, P: fmt::Pointer> fmt::Pointer for Pin<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.pointer, f)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Pin<&'a U>> for Pin<&'a T> {}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Pin<&'a mut U>> for Pin<&'a mut T> {}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: ?Sized> Unpin for Pin<&'a T> {}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: ?Sized> Unpin for Pin<&'a mut T> {}
