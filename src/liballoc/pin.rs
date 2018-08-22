// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
//! by wrapping the data in pinning pointer types, such as [`PinMut`] and [`PinBox`],
//! which are otherwise equivalent to `& mut` and [`Box`], respectively.
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
//! [`PinMut`]: struct.PinMut.html
//! [`PinBox`]: struct.PinBox.html
//! [`Unpin`]: trait.Unpin.html
//! [`swap`]: ../../std/mem/fn.swap.html
//! [`Box`]: ../boxed/struct.Box.html
//!
//! # Examples
//!
//! ```rust
//! #![feature(pin)]
//!
//! use std::pin::PinBox;
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
//!     fn new(data: String) -> PinBox<Self> {
//!         let res = Unmovable {
//!             data,
//!             // we only create the pointer once the data is in place
//!             // otherwise it will have already moved before we even started
//!             slice: NonNull::dangling(),
//!             _pin: Pinned,
//!         };
//!         let mut boxed = PinBox::new(res);
//!
//!         let slice = NonNull::from(&boxed.data);
//!         // we know this is safe because modifying a field doesn't move the whole struct
//!         unsafe { PinBox::get_mut(&mut boxed).slice = slice };
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

pub use core::pin::*;
pub use core::marker::Unpin;

use core::convert::From;
use core::fmt;
use core::future::{Future, FutureObj, LocalFutureObj, UnsafeFutureObj};
use core::marker::Unsize;
use core::ops::{CoerceUnsized, Deref, DerefMut};
use core::task::{Context, Poll};

use boxed::Box;

/// A pinned, heap allocated reference.
///
/// This type is similar to [`Box`], except that it pins its value,
/// which prevents it from moving out of the reference, unless it implements [`Unpin`].
///
/// See the [module documentation] for furthur explaination on pinning.
///
/// [`Box`]: ../boxed/struct.Box.html
/// [`Unpin`]: ../../std/marker/trait.Unpin.html
/// [module documentation]: index.html
#[unstable(feature = "pin", issue = "49150")]
#[fundamental]
#[repr(transparent)]
pub struct PinBox<T: ?Sized> {
    inner: Box<T>,
}

#[unstable(feature = "pin", issue = "49150")]
impl<T> PinBox<T> {
    /// Allocate memory on the heap, move the data into it and pin it.
    #[unstable(feature = "pin", issue = "49150")]
    pub fn new(data: T) -> PinBox<T> {
        PinBox { inner: Box::new(data) }
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: ?Sized> PinBox<T> {
    /// Get a pinned reference to the data in this PinBox.
    #[inline]
    pub fn as_pin_mut<'a>(&'a mut self) -> PinMut<'a, T> {
        unsafe { PinMut::new_unchecked(&mut *self.inner) }
    }

    /// Constructs a `PinBox` from a raw pointer.
    ///
    /// After calling this function, the raw pointer is owned by the
    /// resulting `PinBox`. Specifically, the `PinBox` destructor will call
    /// the destructor of `T` and free the allocated memory. Since the
    /// way `PinBox` allocates and releases memory is unspecified, the
    /// only valid pointer to pass to this function is the one taken
    /// from another `PinBox` via the [`PinBox::into_raw`] function.
    ///
    /// This function is unsafe because improper use may lead to
    /// memory problems. For example, a double-free may occur if the
    /// function is called twice on the same raw pointer.
    ///
    /// [`PinBox::into_raw`]: struct.PinBox.html#method.into_raw
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(pin)]
    /// use std::pin::PinBox;
    /// let x = PinBox::new(5);
    /// let ptr = PinBox::into_raw(x);
    /// let x = unsafe { PinBox::from_raw(ptr) };
    /// ```
    #[inline]
    pub unsafe fn from_raw(raw: *mut T) -> Self {
        PinBox { inner: Box::from_raw(raw) }
    }

    /// Consumes the `PinBox`, returning the wrapped raw pointer.
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `PinBox`. In particular, the
    /// caller should properly destroy `T` and release the memory. The
    /// proper way to do so is to convert the raw pointer back into a
    /// `PinBox` with the [`PinBox::from_raw`] function.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `PinBox::into_raw(b)` instead of `b.into_raw()`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// [`PinBox::from_raw`]: struct.PinBox.html#method.from_raw
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(pin)]
    /// use std::pin::PinBox;
    /// let x = PinBox::new(5);
    /// let ptr = PinBox::into_raw(x);
    /// ```
    #[inline]
    pub fn into_raw(b: PinBox<T>) -> *mut T {
        Box::into_raw(b.inner)
    }

    /// Get a mutable reference to the data inside this PinBox.
    ///
    /// This function is unsafe. Users must guarantee that the data is never
    /// moved out of this reference.
    #[inline]
    pub unsafe fn get_mut<'a>(this: &'a mut PinBox<T>) -> &'a mut T {
        &mut *this.inner
    }

    /// Convert this PinBox into an unpinned Box.
    ///
    /// This function is unsafe. Users must guarantee that the data is never
    /// moved out of the box.
    #[inline]
    pub unsafe fn unpin(this: PinBox<T>) -> Box<T> {
        this.inner
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: ?Sized> From<Box<T>> for PinBox<T> {
    fn from(boxed: Box<T>) -> PinBox<T> {
        PinBox { inner: boxed }
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: ?Sized> Deref for PinBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &*self.inner
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: Unpin + ?Sized> DerefMut for PinBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut *self.inner
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: fmt::Display + ?Sized> fmt::Display for PinBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&*self.inner, f)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: fmt::Debug + ?Sized> fmt::Debug for PinBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&*self.inner, f)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: ?Sized> fmt::Pointer for PinBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // It's not possible to extract the inner Uniq directly from the Box,
        // instead we cast it to a *const which aliases the Unique
        let ptr: *const T = &*self.inner;
        fmt::Pointer::fmt(&ptr, f)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<PinBox<U>> for PinBox<T> {}

#[unstable(feature = "pin", issue = "49150")]
impl<T: ?Sized> Unpin for PinBox<T> {}

#[unstable(feature = "futures_api", issue = "50547")]
impl<F: ?Sized + Future> Future for PinBox<F> {
    type Output = F::Output;

    fn poll(mut self: PinMut<Self>, cx: &mut Context) -> Poll<Self::Output> {
        self.as_pin_mut().poll(cx)
    }
}

#[unstable(feature = "futures_api", issue = "50547")]
unsafe impl<'a, T, F> UnsafeFutureObj<'a, T> for PinBox<F>
    where F: Future<Output = T> + 'a
{
    fn into_raw(self) -> *mut () {
        PinBox::into_raw(self) as *mut ()
    }

    unsafe fn poll(ptr: *mut (), cx: &mut Context) -> Poll<T> {
        let ptr = ptr as *mut F;
        let pin: PinMut<F> = PinMut::new_unchecked(&mut *ptr);
        pin.poll(cx)
    }

    unsafe fn drop(ptr: *mut ()) {
        drop(PinBox::from_raw(ptr as *mut F))
    }
}

#[unstable(feature = "futures_api", issue = "50547")]
impl<'a, F: Future<Output = ()> + Send + 'a> From<PinBox<F>> for FutureObj<'a, ()> {
    fn from(boxed: PinBox<F>) -> Self {
        FutureObj::new(boxed)
    }
}

#[unstable(feature = "futures_api", issue = "50547")]
impl<'a, F: Future<Output = ()> + 'a> From<PinBox<F>> for LocalFutureObj<'a, ()> {
    fn from(boxed: PinBox<F>) -> Self {
        LocalFutureObj::new(boxed)
    }
}
