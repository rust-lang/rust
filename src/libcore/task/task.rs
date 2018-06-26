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
use mem::PinMut;
use super::{Context, Poll};

/// A custom trait object for polling tasks, roughly akin to
/// `Box<Future<Output = ()> + Send>`.
pub struct TaskObj {
    ptr: *mut (),
    poll_fn: unsafe fn(*mut (), &mut Context) -> Poll<()>,
    drop_fn: unsafe fn(*mut ()),
}

unsafe impl Send for TaskObj {}

impl TaskObj {
    /// Create a `TaskObj` from a custom trait object representation.
    #[inline]
    pub fn new<T: UnsafeTask>(t: T) -> TaskObj {
        TaskObj {
            ptr: t.into_raw(),
            poll_fn: T::poll,
            drop_fn: T::drop,
        }
    }
}

impl fmt::Debug for TaskObj {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("TaskObj")
            .finish()
    }
}

impl Future for TaskObj {
    type Output = ();

    #[inline]
    fn poll(self: PinMut<Self>, cx: &mut Context) -> Poll<()> {
        unsafe {
            (self.poll_fn)(self.ptr, cx)
        }
    }
}

impl Drop for TaskObj {
    fn drop(&mut self) {
        unsafe {
            (self.drop_fn)(self.ptr)
        }
    }
}

/// A custom implementation of a task trait object for `TaskObj`, providing
/// a hand-rolled vtable.
///
/// This custom representation is typically used only in `no_std` contexts,
/// where the default `Box`-based implementation is not available.
///
/// The implementor must guarantee that it is safe to call `poll` repeatedly (in
/// a non-concurrent fashion) with the result of `into_raw` until `drop` is
/// called.
pub unsafe trait UnsafeTask: Send + 'static {
    /// Convert a owned instance into a (conceptually owned) void pointer.
    fn into_raw(self) -> *mut ();

    /// Poll the task represented by the given void pointer.
    ///
    /// # Safety
    ///
    /// The trait implementor must guarantee that it is safe to repeatedly call
    /// `poll` with the result of `into_raw` until `drop` is called; such calls
    /// are not, however, allowed to race with each other or with calls to `drop`.
    unsafe fn poll(task: *mut (), cx: &mut Context) -> Poll<()>;

    /// Drops the task represented by the given void pointer.
    ///
    /// # Safety
    ///
    /// The trait implementor must guarantee that it is safe to call this
    /// function once per `into_raw` invocation; that call cannot race with
    /// other calls to `drop` or `poll`.
    unsafe fn drop(task: *mut ());
}
