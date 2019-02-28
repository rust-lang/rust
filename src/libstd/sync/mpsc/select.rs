//! Selection over an array of receivers
//!
//! This module contains the implementation machinery necessary for selecting
//! over a number of receivers. One large goal of this module is to provide an
//! efficient interface to selecting over any receiver of any type.
//!
//! This is achieved through an architecture of a "receiver set" in which
//! receivers are added to a set and then the entire set is waited on at once.
//! The set can be waited on multiple times to prevent re-adding each receiver
//! to the set.
//!
//! Usage of this module is currently encouraged to go through the use of the
//! `select!` macro. This macro allows naturally binding of variables to the
//! received values of receivers in a much more natural syntax then usage of the
//! `Select` structure directly.
//!
//! # Examples
//!
//! ```rust
//! #![feature(mpsc_select)]
//!
//! use std::sync::mpsc::channel;
//!
//! let (tx1, rx1) = channel();
//! let (tx2, rx2) = channel();
//!
//! tx1.send(1).unwrap();
//! tx2.send(2).unwrap();
//!
//! select! {
//!     val = rx1.recv() => {
//!         assert_eq!(val.unwrap(), 1);
//!     },
//!     val = rx2.recv() => {
//!         assert_eq!(val.unwrap(), 2);
//!     }
//! }
//! ```

#![allow(dead_code)]
#![unstable(feature = "mpsc_select",
            reason = "This implementation, while likely sufficient, is unsafe and \
                      likely to be error prone. At some point in the future this \
                      module will be removed.",
            issue = "27800")]
#![rustc_deprecated(since = "1.32.0",
                    reason = "channel selection will be removed in a future release")]

use core::cell::{Cell, UnsafeCell};
use core::marker;
use core::ptr;
use core::usize;

use crate::fmt;
use crate::sync::mpsc::{Receiver, RecvError};
use crate::sync::mpsc::blocking::{self, SignalToken};

/// The "receiver set" of the select interface. This structure is used to manage
/// a set of receivers which are being selected over.
pub struct Select {
    inner: UnsafeCell<SelectInner>,
    next_id: Cell<usize>,
}

struct SelectInner {
    head: *mut Handle<'static, ()>,
    tail: *mut Handle<'static, ()>,
}

impl !marker::Send for Select {}

/// A handle to a receiver which is currently a member of a `Select` set of
/// receivers. This handle is used to keep the receiver in the set as well as
/// interact with the underlying receiver.
pub struct Handle<'rx, T:Send+'rx> {
    /// The ID of this handle, used to compare against the return value of
    /// `Select::wait()`.
    id: usize,
    selector: *mut SelectInner,
    next: *mut Handle<'static, ()>,
    prev: *mut Handle<'static, ()>,
    added: bool,
    packet: &'rx (dyn Packet+'rx),

    // due to our fun transmutes, we be sure to place this at the end. (nothing
    // previous relies on T)
    rx: &'rx Receiver<T>,
}

struct Packets { cur: *mut Handle<'static, ()> }

#[doc(hidden)]
#[derive(PartialEq, Eq)]
pub enum StartResult {
    Installed,
    Abort,
}

#[doc(hidden)]
pub trait Packet {
    fn can_recv(&self) -> bool;
    fn start_selection(&self, token: SignalToken) -> StartResult;
    fn abort_selection(&self) -> bool;
}

impl Select {
    /// Creates a new selection structure. This set is initially empty.
    ///
    /// Usage of this struct directly can sometimes be burdensome, and usage is much easier through
    /// the `select!` macro.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(mpsc_select)]
    ///
    /// use std::sync::mpsc::Select;
    ///
    /// let select = Select::new();
    /// ```
    pub fn new() -> Select {
        Select {
            inner: UnsafeCell::new(SelectInner {
                head: ptr::null_mut(),
                tail: ptr::null_mut(),
            }),
            next_id: Cell::new(1),
        }
    }

    /// Creates a new handle into this receiver set for a new receiver. Note
    /// that this does *not* add the receiver to the receiver set, for that you
    /// must call the `add` method on the handle itself.
    pub fn handle<'a, T: Send>(&'a self, rx: &'a Receiver<T>) -> Handle<'a, T> {
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        Handle {
            id,
            selector: self.inner.get(),
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
            added: false,
            rx,
            packet: rx,
        }
    }

    /// Waits for an event on this receiver set. The returned value is *not* an
    /// index, but rather an ID. This ID can be queried against any active
    /// `Handle` structures (each one has an `id` method). The handle with
    /// the matching `id` will have some sort of event available on it. The
    /// event could either be that data is available or the corresponding
    /// channel has been closed.
    pub fn wait(&self) -> usize {
        self.wait2(true)
    }

    /// Helper method for skipping the preflight checks during testing
    pub(super) fn wait2(&self, do_preflight_checks: bool) -> usize {
        // Note that this is currently an inefficient implementation. We in
        // theory have knowledge about all receivers in the set ahead of time,
        // so this method shouldn't really have to iterate over all of them yet
        // again. The idea with this "receiver set" interface is to get the
        // interface right this time around, and later this implementation can
        // be optimized.
        //
        // This implementation can be summarized by:
        //
        //      fn select(receivers) {
        //          if any receiver ready { return ready index }
        //          deschedule {
        //              block on all receivers
        //          }
        //          unblock on all receivers
        //          return ready index
        //      }
        //
        // Most notably, the iterations over all of the receivers shouldn't be
        // necessary.
        unsafe {
            // Stage 1: preflight checks. Look for any packets ready to receive
            if do_preflight_checks {
                for handle in self.iter() {
                    if (*handle).packet.can_recv() {
                        return (*handle).id();
                    }
                }
            }

            // Stage 2: begin the blocking process
            //
            // Create a number of signal tokens, and install each one
            // sequentially until one fails. If one fails, then abort the
            // selection on the already-installed tokens.
            let (wait_token, signal_token) = blocking::tokens();
            for (i, handle) in self.iter().enumerate() {
                match (*handle).packet.start_selection(signal_token.clone()) {
                    StartResult::Installed => {}
                    StartResult::Abort => {
                        // Go back and abort the already-begun selections
                        for handle in self.iter().take(i) {
                            (*handle).packet.abort_selection();
                        }
                        return (*handle).id;
                    }
                }
            }

            // Stage 3: no messages available, actually block
            wait_token.wait();

            // Stage 4: there *must* be message available; find it.
            //
            // Abort the selection process on each receiver. If the abort
            // process returns `true`, then that means that the receiver is
            // ready to receive some data. Note that this also means that the
            // receiver may have yet to have fully read the `to_wake` field and
            // woken us up (although the wakeup is guaranteed to fail).
            //
            // This situation happens in the window of where a sender invokes
            // increment(), sees -1, and then decides to wake up the thread. After
            // all this is done, the sending thread will set `selecting` to
            // `false`. Until this is done, we cannot return. If we were to
            // return, then a sender could wake up a receiver which has gone
            // back to sleep after this call to `select`.
            //
            // Note that it is a "fairly small window" in which an increment()
            // views that it should wake a thread up until the `selecting` bit
            // is set to false. For now, the implementation currently just spins
            // in a yield loop. This is very distasteful, but this
            // implementation is already nowhere near what it should ideally be.
            // A rewrite should focus on avoiding a yield loop, and for now this
            // implementation is tying us over to a more efficient "don't
            // iterate over everything every time" implementation.
            let mut ready_id = usize::MAX;
            for handle in self.iter() {
                if (*handle).packet.abort_selection() {
                    ready_id = (*handle).id;
                }
            }

            // We must have found a ready receiver
            assert!(ready_id != usize::MAX);
            return ready_id;
        }
    }

    fn iter(&self) -> Packets { Packets { cur: unsafe { &*self.inner.get() }.head } }
}

impl<'rx, T: Send> Handle<'rx, T> {
    /// Retrieves the ID of this handle.
    #[inline]
    pub fn id(&self) -> usize { self.id }

    /// Blocks to receive a value on the underlying receiver, returning `Some` on
    /// success or `None` if the channel disconnects. This function has the same
    /// semantics as `Receiver.recv`
    pub fn recv(&mut self) -> Result<T, RecvError> { self.rx.recv() }

    /// Adds this handle to the receiver set that the handle was created from. This
    /// method can be called multiple times, but it has no effect if `add` was
    /// called previously.
    ///
    /// This method is unsafe because it requires that the `Handle` is not moved
    /// while it is added to the `Select` set.
    pub unsafe fn add(&mut self) {
        if self.added { return }
        let selector = &mut *self.selector;
        let me = self as *mut Handle<'rx, T> as *mut Handle<'static, ()>;

        if selector.head.is_null() {
            selector.head = me;
            selector.tail = me;
        } else {
            (*me).prev = selector.tail;
            assert!((*me).next.is_null());
            (*selector.tail).next = me;
            selector.tail = me;
        }
        self.added = true;
    }

    /// Removes this handle from the `Select` set. This method is unsafe because
    /// it has no guarantee that the `Handle` was not moved since `add` was
    /// called.
    pub unsafe fn remove(&mut self) {
        if !self.added { return }

        let selector = &mut *self.selector;
        let me = self as *mut Handle<'rx, T> as *mut Handle<'static, ()>;

        if self.prev.is_null() {
            assert_eq!(selector.head, me);
            selector.head = self.next;
        } else {
            (*self.prev).next = self.next;
        }
        if self.next.is_null() {
            assert_eq!(selector.tail, me);
            selector.tail = self.prev;
        } else {
            (*self.next).prev = self.prev;
        }

        self.next = ptr::null_mut();
        self.prev = ptr::null_mut();

        self.added = false;
    }
}

impl Drop for Select {
    fn drop(&mut self) {
        unsafe {
            assert!((&*self.inner.get()).head.is_null());
            assert!((&*self.inner.get()).tail.is_null());
        }
    }
}

impl<T: Send> Drop for Handle<'_, T> {
    fn drop(&mut self) {
        unsafe { self.remove() }
    }
}

impl Iterator for Packets {
    type Item = *mut Handle<'static, ()>;

    fn next(&mut self) -> Option<*mut Handle<'static, ()>> {
        if self.cur.is_null() {
            None
        } else {
            let ret = Some(self.cur);
            unsafe { self.cur = (*self.cur).next; }
            ret
        }
    }
}

impl fmt::Debug for Select {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Select").finish()
    }
}

impl<T: Send> fmt::Debug for Handle<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Handle").finish()
    }
}
