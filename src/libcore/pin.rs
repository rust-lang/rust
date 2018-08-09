// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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
//! see the [standard library module] for more information
//!
//! [standard library module]: ../../std/pin/index.html

#![unstable(feature = "pin", issue = "49150")]

use fmt;
use future::{Future, UnsafeFutureObj};
use marker::{Sized, Unpin, Unsize};
use task::{Context, Poll};
use ops::{Deref, DerefMut, CoerceUnsized};

/// A pinned reference.
///
/// A pinned reference is a lot like a mutable reference, except that it is not
/// safe to move a value out of a pinned reference unless the type of that
/// value implements the `Unpin` trait.
#[unstable(feature = "pin", issue = "49150")]
#[fundamental]
pub struct PinMut<'a, T: ?Sized + 'a> {
    inner: &'a mut T,
}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: ?Sized + Unpin> PinMut<'a, T> {
    /// Construct a new `PinMut` around a reference to some data of a type that
    /// implements `Unpin`.
    #[unstable(feature = "pin", issue = "49150")]
    pub fn new(reference: &'a mut T) -> PinMut<'a, T> {
        PinMut { inner: reference }
    }

    /// Get a mutable reference to the data inside of this `PinMut`.
    #[unstable(feature = "pin", issue = "49150")]
    pub fn get_mut(this: PinMut<'a, T>) -> &'a mut T {
        this.inner
    }
}


#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: ?Sized> PinMut<'a, T> {
    /// Construct a new `PinMut` around a reference to some data of a type that
    /// may or may not implement `Unpin`.
    ///
    /// This constructor is unsafe because we do not know what will happen with
    /// that data after the reference ends. If you cannot guarantee that the
    /// data will never move again, calling this constructor is invalid.
    #[unstable(feature = "pin", issue = "49150")]
    pub unsafe fn new_unchecked(reference: &'a mut T) -> PinMut<'a, T> {
        PinMut { inner: reference }
    }

    /// Reborrow a `PinMut` for a shorter lifetime.
    ///
    /// For example, `PinMut::get_mut(x.reborrow())` (unsafely) returns a
    /// short-lived mutable reference reborrowing from `x`.
    #[unstable(feature = "pin", issue = "49150")]
    pub fn reborrow<'b>(&'b mut self) -> PinMut<'b, T> {
        PinMut { inner: self.inner }
    }

    /// Get a mutable reference to the data inside of this `PinMut`.
    ///
    /// This function is unsafe. You must guarantee that you will never move
    /// the data out of the mutable reference you receive when you call this
    /// function.
    #[unstable(feature = "pin", issue = "49150")]
    pub unsafe fn get_mut_unchecked(this: PinMut<'a, T>) -> &'a mut T {
        this.inner
    }

    /// Construct a new pin by mapping the interior value.
    ///
    /// For example, if you  wanted to get a `PinMut` of a field of something,
    /// you could use this to get access to that field in one line of code.
    ///
    /// This function is unsafe. You must guarantee that the data you return
    /// will not move so long as the argument value does not move (for example,
    /// because it is one of the fields of that value), and also that you do
    /// not move out of the argument you receive to the interior function.
    #[unstable(feature = "pin", issue = "49150")]
    pub unsafe fn map_unchecked<U, F>(this: PinMut<'a, T>, f: F) -> PinMut<'a, U> where
        F: FnOnce(&mut T) -> &mut U
    {
        PinMut { inner: f(this.inner) }
    }

    /// Assign a new value to the memory behind the pinned reference.
    #[unstable(feature = "pin", issue = "49150")]
    pub fn set(this: PinMut<'a, T>, value: T)
        where T: Sized,
    {
        *this.inner = value;
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: ?Sized> Deref for PinMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &*self.inner
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: ?Sized + Unpin> DerefMut for PinMut<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.inner
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: fmt::Debug + ?Sized> fmt::Debug for PinMut<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: fmt::Display + ?Sized> fmt::Display for PinMut<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: ?Sized> fmt::Pointer for PinMut<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&(&*self.inner as *const T), f)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<PinMut<'a, U>> for PinMut<'a, T> {}

#[unstable(feature = "pin", issue = "49150")]
impl<'a, T: ?Sized> Unpin for PinMut<'a, T> {}

#[unstable(feature = "futures_api", issue = "50547")]
unsafe impl<'a, T, F> UnsafeFutureObj<'a, T> for PinMut<'a, F>
    where F: Future<Output = T> + 'a
{
    fn into_raw(self) -> *mut () {
        unsafe { PinMut::get_mut_unchecked(self) as *mut F as *mut () }
    }

    unsafe fn poll(ptr: *mut (), cx: &mut Context) -> Poll<T> {
        PinMut::new_unchecked(&mut *(ptr as *mut F)).poll(cx)
    }

    unsafe fn drop(_ptr: *mut ()) {}
}
