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

//! Asynchronous values.

use mem::PinMut;
use marker::Unpin;
use task::{self, Poll};

/// A future represents an asychronous computation.
///
/// A future is a value that may not have finished computing yet. This kind of
/// "asynchronous value" makes it possible for a thread to continue doing useful
/// work while it waits for the value to become available.
///
/// # The `poll` method
///
/// The core method of future, `poll`, *attempts* to resolve the future into a
/// final value. This method does not block if the value is not ready. Instead,
/// the current task is scheduled to be woken up when it's possible to make
/// further progress by `poll`ing again. The wake up is performed using
/// `cx.waker()`, a handle for waking up the current task.
///
/// When using a future, you generally won't call `poll` directly, but instead
/// `await!` the value.
pub trait Future {
    /// The result of the `Future`.
    type Output;

    /// Attempt to resolve the future to a final value, registering
    /// the current task for wakeup if the value is not yet available.
    ///
    /// # Return value
    ///
    /// This function returns:
    ///
    /// - `Poll::Pending` if the future is not ready yet
    /// - `Poll::Ready(val)` with the result `val` of this future if it finished
    /// successfully.
    ///
    /// Once a future has finished, clients should not `poll` it again.
    ///
    /// When a future is not ready yet, `poll` returns
    /// [`Poll::Pending`](::task::Poll). The future will *also* register the
    /// interest of the current task in the value being produced. For example,
    /// if the future represents the availability of data on a socket, then the
    /// task is recorded so that when data arrives, it is woken up (via
    /// [`cx.waker()`](::task::Context::waker)). Once a task has been woken up,
    /// it should attempt to `poll` the future again, which may or may not
    /// produce a final value.
    ///
    /// Note that if `Pending` is returned it only means that the *current* task
    /// (represented by the argument `cx`) will receive a notification. Tasks
    /// from previous calls to `poll` will *not* receive notifications.
    ///
    /// # Runtime characteristics
    ///
    /// Futures alone are *inert*; they must be *actively* `poll`ed to make
    /// progress, meaning that each time the current task is woken up, it should
    /// actively re-`poll` pending futures that it still has an interest in.
    ///
    /// The `poll` function is not called repeatedly in a tight loop for
    /// futures, but only whenever the future itself is ready, as signaled via
    /// the `Waker` inside `task::Context`. If you're familiar with the
    /// `poll(2)` or `select(2)` syscalls on Unix it's worth noting that futures
    /// typically do *not* suffer the same problems of "all wakeups must poll
    /// all events"; they are more like `epoll(4)`.
    ///
    /// An implementation of `poll` should strive to return quickly, and must
    /// *never* block. Returning quickly prevents unnecessarily clogging up
    /// threads or event loops. If it is known ahead of time that a call to
    /// `poll` may end up taking awhile, the work should be offloaded to a
    /// thread pool (or something similar) to ensure that `poll` can return
    /// quickly.
    ///
    /// # Panics
    ///
    /// Once a future has completed (returned `Ready` from `poll`),
    /// then any future calls to `poll` may panic, block forever, or otherwise
    /// cause bad behavior. The `Future` trait itself provides no guarantees
    /// about the behavior of `poll` after a future has completed.
    fn poll(self: PinMut<Self>, cx: &mut task::Context) -> Poll<Self::Output>;
}

impl<'a, F: ?Sized + Future + Unpin> Future for &'a mut F {
    type Output = F::Output;

    fn poll(mut self: PinMut<Self>, cx: &mut task::Context) -> Poll<Self::Output> {
        F::poll(PinMut::new(&mut **self), cx)
    }
}

impl<'a, F: ?Sized + Future> Future for PinMut<'a, F> {
    type Output = F::Output;

    fn poll(mut self: PinMut<Self>, cx: &mut task::Context) -> Poll<Self::Output> {
        F::poll((*self).reborrow(), cx)
    }
}
