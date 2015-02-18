// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
//! # Example
//!
//! ```rust
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
#![unstable(feature = "std_misc",
            reason = "This implementation, while likely sufficient, is unsafe and \
                      likely to be error prone. At some point in the future this \
                      module will likely be replaced, and it is currently \
                      unknown how much API breakage that will cause. The ability \
                      to select over a number of channels will remain forever, \
                      but no guarantees beyond this are being made")]


use core::prelude::*;

use core::cell::Cell;
use core::marker;
use core::mem;
use core::ptr;
use core::usize;

use sync::mpsc::{Receiver, RecvError};
use sync::mpsc::blocking::{self, SignalToken};

/// The "receiver set" of the select interface. This structure is used to manage
/// a set of receivers which are being selected over.
pub struct Select {
    head: *mut Handle<'static, ()>,
    tail: *mut Handle<'static, ()>,
    next_id: Cell<uint>,
}

impl !marker::Send for Select {}

/// A handle to a receiver which is currently a member of a `Select` set of
/// receivers.  This handle is used to keep the receiver in the set as well as
/// interact with the underlying receiver.
pub struct Handle<'rx, T:'rx> {
    /// The ID of this handle, used to compare against the return value of
    /// `Select::wait()`
    id: uint,
    selector: &'rx Select,
    next: *mut Handle<'static, ()>,
    prev: *mut Handle<'static, ()>,
    added: bool,
    packet: &'rx (Packet+'rx),

    // due to our fun transmutes, we be sure to place this at the end. (nothing
    // previous relies on T)
    rx: &'rx Receiver<T>,
}

struct Packets { cur: *mut Handle<'static, ()> }

#[doc(hidden)]
#[derive(PartialEq)]
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
    /// use std::sync::mpsc::Select;
    ///
    /// let select = Select::new();
    /// ```
    pub fn new() -> Select {
        Select {
            head: ptr::null_mut(),
            tail: ptr::null_mut(),
            next_id: Cell::new(1),
        }
    }

    /// Creates a new handle into this receiver set for a new receiver. Note
    /// that this does *not* add the receiver to the receiver set, for that you
    /// must call the `add` method on the handle itself.
    pub fn handle<'a, T: Send + 'static>(&'a self, rx: &'a Receiver<T>) -> Handle<'a, T> {
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        Handle {
            id: id,
            selector: self,
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
            added: false,
            rx: rx,
            packet: rx,
        }
    }

    /// Waits for an event on this receiver set. The returned value is *not* an
    /// index, but rather an id. This id can be queried against any active
    /// `Handle` structures (each one has an `id` method). The handle with
    /// the matching `id` will have some sort of event available on it. The
    /// event could either be that data is available or the corresponding
    /// channel has been closed.
    pub fn wait(&self) -> uint {
        self.wait2(true)
    }

    /// Helper method for skipping the preflight checks during testing
    fn wait2(&self, do_preflight_checks: bool) -> uint {
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
            // increment(), sees -1, and then decides to wake up the task. After
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

    fn iter(&self) -> Packets { Packets { cur: self.head } }
}

impl<'rx, T: Send + 'static> Handle<'rx, T> {
    /// Retrieve the id of this handle.
    #[inline]
    pub fn id(&self) -> uint { self.id }

    /// Block to receive a value on the underlying receiver, returning `Some` on
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
        let selector: &mut Select = mem::transmute(&*self.selector);
        let me: *mut Handle<'static, ()> = mem::transmute(&*self);

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

        let selector: &mut Select = mem::transmute(&*self.selector);
        let me: *mut Handle<'static, ()> = mem::transmute(&*self);

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

#[unsafe_destructor]
impl Drop for Select {
    fn drop(&mut self) {
        assert!(self.head.is_null());
        assert!(self.tail.is_null());
    }
}

#[unsafe_destructor]
impl<'rx, T: Send + 'static> Drop for Handle<'rx, T> {
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

#[cfg(test)]
#[allow(unused_imports)]
mod test {
    use prelude::v1::*;

    use thread;
    use sync::mpsc::*;

    // Don't use the libstd version so we can pull in the right Select structure
    // (std::comm points at the wrong one)
    macro_rules! select {
        (
            $($name:pat = $rx:ident.$meth:ident() => $code:expr),+
        ) => ({
            let sel = Select::new();
            $( let mut $rx = sel.handle(&$rx); )+
            unsafe {
                $( $rx.add(); )+
            }
            let ret = sel.wait();
            $( if ret == $rx.id() { let $name = $rx.$meth(); $code } else )+
            { unreachable!() }
        })
    }

    #[test]
    fn smoke() {
        let (tx1, rx1) = channel::<int>();
        let (tx2, rx2) = channel::<int>();
        tx1.send(1).unwrap();
        select! {
            foo = rx1.recv() => { assert_eq!(foo.unwrap(), 1); },
            _bar = rx2.recv() => { panic!() }
        }
        tx2.send(2).unwrap();
        select! {
            _foo = rx1.recv() => { panic!() },
            bar = rx2.recv() => { assert_eq!(bar.unwrap(), 2) }
        }
        drop(tx1);
        select! {
            foo = rx1.recv() => { assert!(foo.is_err()); },
            _bar = rx2.recv() => { panic!() }
        }
        drop(tx2);
        select! {
            bar = rx2.recv() => { assert!(bar.is_err()); }
        }
    }

    #[test]
    fn smoke2() {
        let (_tx1, rx1) = channel::<int>();
        let (_tx2, rx2) = channel::<int>();
        let (_tx3, rx3) = channel::<int>();
        let (_tx4, rx4) = channel::<int>();
        let (tx5, rx5) = channel::<int>();
        tx5.send(4).unwrap();
        select! {
            _foo = rx1.recv() => { panic!("1") },
            _foo = rx2.recv() => { panic!("2") },
            _foo = rx3.recv() => { panic!("3") },
            _foo = rx4.recv() => { panic!("4") },
            foo = rx5.recv() => { assert_eq!(foo.unwrap(), 4); }
        }
    }

    #[test]
    fn closed() {
        let (_tx1, rx1) = channel::<int>();
        let (tx2, rx2) = channel::<int>();
        drop(tx2);

        select! {
            _a1 = rx1.recv() => { panic!() },
            a2 = rx2.recv() => { assert!(a2.is_err()); }
        }
    }

    #[test]
    fn unblocks() {
        let (tx1, rx1) = channel::<int>();
        let (_tx2, rx2) = channel::<int>();
        let (tx3, rx3) = channel::<int>();

        let _t = thread::spawn(move|| {
            for _ in 0..20 { thread::yield_now(); }
            tx1.send(1).unwrap();
            rx3.recv().unwrap();
            for _ in 0..20 { thread::yield_now(); }
        });

        select! {
            a = rx1.recv() => { assert_eq!(a.unwrap(), 1); },
            _b = rx2.recv() => { panic!() }
        }
        tx3.send(1).unwrap();
        select! {
            a = rx1.recv() => { assert!(a.is_err()) },
            _b = rx2.recv() => { panic!() }
        }
    }

    #[test]
    fn both_ready() {
        let (tx1, rx1) = channel::<int>();
        let (tx2, rx2) = channel::<int>();
        let (tx3, rx3) = channel::<()>();

        let _t = thread::spawn(move|| {
            for _ in 0..20 { thread::yield_now(); }
            tx1.send(1).unwrap();
            tx2.send(2).unwrap();
            rx3.recv().unwrap();
        });

        select! {
            a = rx1.recv() => { assert_eq!(a.unwrap(), 1); },
            a = rx2.recv() => { assert_eq!(a.unwrap(), 2); }
        }
        select! {
            a = rx1.recv() => { assert_eq!(a.unwrap(), 1); },
            a = rx2.recv() => { assert_eq!(a.unwrap(), 2); }
        }
        assert_eq!(rx1.try_recv(), Err(TryRecvError::Empty));
        assert_eq!(rx2.try_recv(), Err(TryRecvError::Empty));
        tx3.send(()).unwrap();
    }

    #[test]
    fn stress() {
        static AMT: int = 10000;
        let (tx1, rx1) = channel::<int>();
        let (tx2, rx2) = channel::<int>();
        let (tx3, rx3) = channel::<()>();

        let _t = thread::spawn(move|| {
            for i in 0..AMT {
                if i % 2 == 0 {
                    tx1.send(i).unwrap();
                } else {
                    tx2.send(i).unwrap();
                }
                rx3.recv().unwrap();
            }
        });

        for i in 0..AMT {
            select! {
                i1 = rx1.recv() => { assert!(i % 2 == 0 && i == i1.unwrap()); },
                i2 = rx2.recv() => { assert!(i % 2 == 1 && i == i2.unwrap()); }
            }
            tx3.send(()).unwrap();
        }
    }

    #[test]
    fn cloning() {
        let (tx1, rx1) = channel::<int>();
        let (_tx2, rx2) = channel::<int>();
        let (tx3, rx3) = channel::<()>();

        let _t = thread::spawn(move|| {
            rx3.recv().unwrap();
            tx1.clone();
            assert_eq!(rx3.try_recv(), Err(TryRecvError::Empty));
            tx1.send(2).unwrap();
            rx3.recv().unwrap();
        });

        tx3.send(()).unwrap();
        select! {
            _i1 = rx1.recv() => {},
            _i2 = rx2.recv() => panic!()
        }
        tx3.send(()).unwrap();
    }

    #[test]
    fn cloning2() {
        let (tx1, rx1) = channel::<int>();
        let (_tx2, rx2) = channel::<int>();
        let (tx3, rx3) = channel::<()>();

        let _t = thread::spawn(move|| {
            rx3.recv().unwrap();
            tx1.clone();
            assert_eq!(rx3.try_recv(), Err(TryRecvError::Empty));
            tx1.send(2).unwrap();
            rx3.recv().unwrap();
        });

        tx3.send(()).unwrap();
        select! {
            _i1 = rx1.recv() => {},
            _i2 = rx2.recv() => panic!()
        }
        tx3.send(()).unwrap();
    }

    #[test]
    fn cloning3() {
        let (tx1, rx1) = channel::<()>();
        let (tx2, rx2) = channel::<()>();
        let (tx3, rx3) = channel::<()>();
        let _t = thread::spawn(move|| {
            let s = Select::new();
            let mut h1 = s.handle(&rx1);
            let mut h2 = s.handle(&rx2);
            unsafe { h2.add(); }
            unsafe { h1.add(); }
            assert_eq!(s.wait(), h2.id);
            tx3.send(()).unwrap();
        });

        for _ in 0..1000 { thread::yield_now(); }
        drop(tx1.clone());
        tx2.send(()).unwrap();
        rx3.recv().unwrap();
    }

    #[test]
    fn preflight1() {
        let (tx, rx) = channel();
        tx.send(()).unwrap();
        select! {
            _n = rx.recv() => {}
        }
    }

    #[test]
    fn preflight2() {
        let (tx, rx) = channel();
        tx.send(()).unwrap();
        tx.send(()).unwrap();
        select! {
            _n = rx.recv() => {}
        }
    }

    #[test]
    fn preflight3() {
        let (tx, rx) = channel();
        drop(tx.clone());
        tx.send(()).unwrap();
        select! {
            _n = rx.recv() => {}
        }
    }

    #[test]
    fn preflight4() {
        let (tx, rx) = channel();
        tx.send(()).unwrap();
        let s = Select::new();
        let mut h = s.handle(&rx);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    }

    #[test]
    fn preflight5() {
        let (tx, rx) = channel();
        tx.send(()).unwrap();
        tx.send(()).unwrap();
        let s = Select::new();
        let mut h = s.handle(&rx);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    }

    #[test]
    fn preflight6() {
        let (tx, rx) = channel();
        drop(tx.clone());
        tx.send(()).unwrap();
        let s = Select::new();
        let mut h = s.handle(&rx);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    }

    #[test]
    fn preflight7() {
        let (tx, rx) = channel::<()>();
        drop(tx);
        let s = Select::new();
        let mut h = s.handle(&rx);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    }

    #[test]
    fn preflight8() {
        let (tx, rx) = channel();
        tx.send(()).unwrap();
        drop(tx);
        rx.recv().unwrap();
        let s = Select::new();
        let mut h = s.handle(&rx);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    }

    #[test]
    fn preflight9() {
        let (tx, rx) = channel();
        drop(tx.clone());
        tx.send(()).unwrap();
        drop(tx);
        rx.recv().unwrap();
        let s = Select::new();
        let mut h = s.handle(&rx);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    }

    #[test]
    fn oneshot_data_waiting() {
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        let _t = thread::spawn(move|| {
            select! {
                _n = rx1.recv() => {}
            }
            tx2.send(()).unwrap();
        });

        for _ in 0..100 { thread::yield_now() }
        tx1.send(()).unwrap();
        rx2.recv().unwrap();
    }

    #[test]
    fn stream_data_waiting() {
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        tx1.send(()).unwrap();
        tx1.send(()).unwrap();
        rx1.recv().unwrap();
        rx1.recv().unwrap();
        let _t = thread::spawn(move|| {
            select! {
                _n = rx1.recv() => {}
            }
            tx2.send(()).unwrap();
        });

        for _ in 0..100 { thread::yield_now() }
        tx1.send(()).unwrap();
        rx2.recv().unwrap();
    }

    #[test]
    fn shared_data_waiting() {
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        drop(tx1.clone());
        tx1.send(()).unwrap();
        rx1.recv().unwrap();
        let _t = thread::spawn(move|| {
            select! {
                _n = rx1.recv() => {}
            }
            tx2.send(()).unwrap();
        });

        for _ in 0..100 { thread::yield_now() }
        tx1.send(()).unwrap();
        rx2.recv().unwrap();
    }

    #[test]
    fn sync1() {
        let (tx, rx) = sync_channel::<int>(1);
        tx.send(1).unwrap();
        select! {
            n = rx.recv() => { assert_eq!(n.unwrap(), 1); }
        }
    }

    #[test]
    fn sync2() {
        let (tx, rx) = sync_channel::<int>(0);
        let _t = thread::spawn(move|| {
            for _ in 0..100 { thread::yield_now() }
            tx.send(1).unwrap();
        });
        select! {
            n = rx.recv() => { assert_eq!(n.unwrap(), 1); }
        }
    }

    #[test]
    fn sync3() {
        let (tx1, rx1) = sync_channel::<int>(0);
        let (tx2, rx2): (Sender<int>, Receiver<int>) = channel();
        let _t = thread::spawn(move|| { tx1.send(1).unwrap(); });
        let _t = thread::spawn(move|| { tx2.send(2).unwrap(); });
        select! {
            n = rx1.recv() => {
                let n = n.unwrap();
                assert_eq!(n, 1);
                assert_eq!(rx2.recv().unwrap(), 2);
            },
            n = rx2.recv() => {
                let n = n.unwrap();
                assert_eq!(n, 2);
                assert_eq!(rx1.recv().unwrap(), 1);
            }
        }
    }
}
