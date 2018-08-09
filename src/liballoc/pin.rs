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

#![unstable(feature = "pin", issue = "49150")]

use core::convert::From;
use core::fmt;
use core::future::{Future, FutureObj, LocalFutureObj, UnsafeFutureObj};
use core::marker::{Unpin, Unsize};
use core::pin::PinMut;
use core::ops::{CoerceUnsized, Deref, DerefMut};
use core::task::{Context, Poll};

use boxed::Box;

/// A pinned, heap allocated reference.
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
impl<T: Unpin + ?Sized> From<PinBox<T>> for Box<T> {
    fn from(pinned: PinBox<T>) -> Box<T> {
        pinned.inner
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
