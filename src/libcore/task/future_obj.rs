// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "futures_api",
            reason = "futures in libcore are unstable",
            issue = "50547")]

use fmt;
use future::Future;
use marker::PhantomData;
use mem::PinMut;
use task::{Context, Poll};

/// A custom trait object for polling futures, roughly akin to
/// `Box<dyn Future<Output = T>>`.
/// Contrary to `FutureObj`, `LocalFutureObj` does not have a `Send` bound.
pub struct LocalFutureObj<T> {
    ptr: *mut (),
    poll_fn: unsafe fn(*mut (), &mut Context) -> Poll<T>,
    drop_fn: unsafe fn(*mut ()),
    _marker: PhantomData<T>,
}

impl<T> LocalFutureObj<T> {
    /// Create a `LocalFutureObj` from a custom trait object representation.
    #[inline]
    pub fn new<F: UnsafeFutureObj<T>>(f: F) -> LocalFutureObj<T> {
        LocalFutureObj {
            ptr: f.into_raw(),
            poll_fn: F::poll,
            drop_fn: F::drop,
            _marker: PhantomData,
        }
    }

    /// Converts the `LocalFutureObj` into a `FutureObj`
    /// To make this operation safe one has to ensure that the `UnsafeFutureObj`
    /// instance from which this `LocalFutureObj` was created actually
    /// implements `Send`.
    #[inline]
    pub unsafe fn as_future_obj(self) -> FutureObj<T> {
        FutureObj(self)
    }
}

impl<T> fmt::Debug for LocalFutureObj<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LocalFutureObj")
            .finish()
    }
}

impl<T> From<FutureObj<T>> for LocalFutureObj<T> {
    #[inline]
    fn from(f: FutureObj<T>) -> LocalFutureObj<T> {
        f.0
    }
}

impl<T> Future for LocalFutureObj<T> {
    type Output = T;

    #[inline]
    fn poll(self: PinMut<Self>, cx: &mut Context) -> Poll<T> {
        unsafe {
            (self.poll_fn)(self.ptr, cx)
        }
    }
}

impl<T> Drop for LocalFutureObj<T> {
    fn drop(&mut self) {
        unsafe {
            (self.drop_fn)(self.ptr)
        }
    }
}

/// A custom trait object for polling futures, roughly akin to
/// `Box<dyn Future<Output = T>> + Send`.
pub struct FutureObj<T>(LocalFutureObj<T>);

unsafe impl<T> Send for FutureObj<T> {}

impl<T> FutureObj<T> {
    /// Create a `FutureObj` from a custom trait object representation.
    #[inline]
    pub fn new<F: UnsafeFutureObj<T> + Send>(f: F) -> FutureObj<T> {
        FutureObj(LocalFutureObj::new(f))
    }
}

impl<T> fmt::Debug for FutureObj<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FutureObj")
            .finish()
    }
}

impl<T> Future for FutureObj<T> {
    type Output = T;

    #[inline]
    fn poll(self: PinMut<Self>, cx: &mut Context) -> Poll<T> {
        let pinned_field = unsafe { PinMut::map_unchecked(self, |x| &mut x.0) };
        pinned_field.poll(cx)
    }
}

/// A custom implementation of a future trait object for `FutureObj`, providing
/// a hand-rolled vtable.
///
/// This custom representation is typically used only in `no_std` contexts,
/// where the default `Box`-based implementation is not available.
///
/// The implementor must guarantee that it is safe to call `poll` repeatedly (in
/// a non-concurrent fashion) with the result of `into_raw` until `drop` is
/// called.
pub unsafe trait UnsafeFutureObj<T>: 'static {
    /// Convert a owned instance into a (conceptually owned) void pointer.
    fn into_raw(self) -> *mut ();

    /// Poll the future represented by the given void pointer.
    ///
    /// # Safety
    ///
    /// The trait implementor must guarantee that it is safe to repeatedly call
    /// `poll` with the result of `into_raw` until `drop` is called; such calls
    /// are not, however, allowed to race with each other or with calls to `drop`.
    unsafe fn poll(future: *mut (), cx: &mut Context) -> Poll<T>;

    /// Drops the future represented by the given void pointer.
    ///
    /// # Safety
    ///
    /// The trait implementor must guarantee that it is safe to call this
    /// function once per `into_raw` invocation; that call cannot race with
    /// other calls to `drop` or `poll`.
    unsafe fn drop(future: *mut ());
}
