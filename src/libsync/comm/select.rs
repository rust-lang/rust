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
//! let (tx1, rx1) = channel();
//! let (tx2, rx2) = channel();
//!
//! tx1.send(1i);
//! tx2.send(2i);
//!
//! select! {
//!     val = rx1.recv() => {
//!         assert_eq!(val, 1i);
//!     },
//!     val = rx2.recv() => {
//!         assert_eq!(val, 2i);
//!     }
//! }
//! ```

#![allow(dead_code)]

use core::prelude::*;

use alloc::owned::Box;
use core::cell::Cell;
use core::kinds::marker;
use core::mem;
use core::uint;
use rustrt::local::Local;
use rustrt::task::{Task, BlockedTask};

use comm::Receiver;

/// The "receiver set" of the select interface. This structure is used to manage
/// a set of receivers which are being selected over.
pub struct Select {
    head: *mut Handle<'static, ()>,
    tail: *mut Handle<'static, ()>,
    next_id: Cell<uint>,
    marker1: marker::NoSend,
}

/// A handle to a receiver which is currently a member of a `Select` set of
/// receivers.  This handle is used to keep the receiver in the set as well as
/// interact with the underlying receiver.
pub struct Handle<'rx, T> {
    /// The ID of this handle, used to compare against the return value of
    /// `Select::wait()`
    id: uint,
    selector: &'rx Select,
    next: *mut Handle<'static, ()>,
    prev: *mut Handle<'static, ()>,
    added: bool,
    packet: &'rx Packet,

    // due to our fun transmutes, we be sure to place this at the end. (nothing
    // previous relies on T)
    rx: &'rx Receiver<T>,
}

struct Packets { cur: *mut Handle<'static, ()> }

#[doc(hidden)]
pub trait Packet {
    fn can_recv(&self) -> bool;
    fn start_selection(&self, task: BlockedTask) -> Result<(), BlockedTask>;
    fn abort_selection(&self) -> bool;
}

impl Select {
    /// Creates a new selection structure. This set is initially empty and
    /// `wait` will fail!() if called.
    ///
    /// Usage of this struct directly can sometimes be burdensome, and usage is
    /// rather much easier through the `select!` macro.
    pub fn new() -> Select {
        Select {
            marker1: marker::NoSend,
            head: 0 as *mut Handle<'static, ()>,
            tail: 0 as *mut Handle<'static, ()>,
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
            id: id,
            selector: self,
            next: 0 as *mut Handle<'static, ()>,
            prev: 0 as *mut Handle<'static, ()>,
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
            let mut amt = 0;
            for p in self.iter() {
                amt += 1;
                if do_preflight_checks && (*p).packet.can_recv() {
                    return (*p).id;
                }
            }
            assert!(amt > 0);

            let mut ready_index = amt;
            let mut ready_id = uint::MAX;
            let mut iter = self.iter().enumerate();

            // Acquire a number of blocking contexts, and block on each one
            // sequentially until one fails. If one fails, then abort
            // immediately so we can go unblock on all the other receivers.
            let task: Box<Task> = Local::take();
            task.deschedule(amt, |task| {
                // Prepare for the block
                let (i, handle) = iter.next().unwrap();
                match (*handle).packet.start_selection(task) {
                    Ok(()) => Ok(()),
                    Err(task) => {
                        ready_index = i;
                        ready_id = (*handle).id;
                        Err(task)
                    }
                }
            });

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
            for handle in self.iter().take(ready_index) {
                if (*handle).packet.abort_selection() {
                    ready_id = (*handle).id;
                }
            }

            assert!(ready_id != uint::MAX);
            return ready_id;
        }
    }

    fn iter(&self) -> Packets { Packets { cur: self.head } }
}

impl<'rx, T: Send> Handle<'rx, T> {
    /// Retrieve the id of this handle.
    #[inline]
    pub fn id(&self) -> uint { self.id }

    /// Receive a value on the underlying receiver. Has the same semantics as
    /// `Receiver.recv`
    pub fn recv(&mut self) -> T { self.rx.recv() }
    /// Block to receive a value on the underlying receiver, returning `Some` on
    /// success or `None` if the channel disconnects. This function has the same
    /// semantics as `Receiver.recv_opt`
    pub fn recv_opt(&mut self) -> Result<T, ()> { self.rx.recv_opt() }

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

        self.next = 0 as *mut Handle<'static, ()>;
        self.prev = 0 as *mut Handle<'static, ()>;

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
impl<'rx, T: Send> Drop for Handle<'rx, T> {
    fn drop(&mut self) {
        unsafe { self.remove() }
    }
}

impl Iterator<*mut Handle<'static, ()>> for Packets {
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
    use std::prelude::*;

    use super::super::*;

    // Don't use the libstd version so we can pull in the right Select structure
    // (std::comm points at the wrong one)
    macro_rules! select {
        (
            $($name:pat = $rx:ident.$meth:ident() => $code:expr),+
        ) => ({
            use comm::Select;
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

    test!(fn smoke() {
        let (tx1, rx1) = channel::<int>();
        let (tx2, rx2) = channel::<int>();
        tx1.send(1);
        select! (
            foo = rx1.recv() => { assert_eq!(foo, 1); },
            _bar = rx2.recv() => { fail!() }
        )
        tx2.send(2);
        select! (
            _foo = rx1.recv() => { fail!() },
            bar = rx2.recv() => { assert_eq!(bar, 2) }
        )
        drop(tx1);
        select! (
            foo = rx1.recv_opt() => { assert_eq!(foo, Err(())); },
            _bar = rx2.recv() => { fail!() }
        )
        drop(tx2);
        select! (
            bar = rx2.recv_opt() => { assert_eq!(bar, Err(())); }
        )
    })

    test!(fn smoke2() {
        let (_tx1, rx1) = channel::<int>();
        let (_tx2, rx2) = channel::<int>();
        let (_tx3, rx3) = channel::<int>();
        let (_tx4, rx4) = channel::<int>();
        let (tx5, rx5) = channel::<int>();
        tx5.send(4);
        select! (
            _foo = rx1.recv() => { fail!("1") },
            _foo = rx2.recv() => { fail!("2") },
            _foo = rx3.recv() => { fail!("3") },
            _foo = rx4.recv() => { fail!("4") },
            foo = rx5.recv() => { assert_eq!(foo, 4); }
        )
    })

    test!(fn closed() {
        let (_tx1, rx1) = channel::<int>();
        let (tx2, rx2) = channel::<int>();
        drop(tx2);

        select! (
            _a1 = rx1.recv_opt() => { fail!() },
            a2 = rx2.recv_opt() => { assert_eq!(a2, Err(())); }
        )
    })

    test!(fn unblocks() {
        let (tx1, rx1) = channel::<int>();
        let (_tx2, rx2) = channel::<int>();
        let (tx3, rx3) = channel::<int>();

        spawn(proc() {
            for _ in range(0u, 20) { task::deschedule(); }
            tx1.send(1);
            rx3.recv();
            for _ in range(0u, 20) { task::deschedule(); }
        });

        select! (
            a = rx1.recv() => { assert_eq!(a, 1); },
            _b = rx2.recv() => { fail!() }
        )
        tx3.send(1);
        select! (
            a = rx1.recv_opt() => { assert_eq!(a, Err(())); },
            _b = rx2.recv() => { fail!() }
        )
    })

    test!(fn both_ready() {
        let (tx1, rx1) = channel::<int>();
        let (tx2, rx2) = channel::<int>();
        let (tx3, rx3) = channel::<()>();

        spawn(proc() {
            for _ in range(0u, 20) { task::deschedule(); }
            tx1.send(1);
            tx2.send(2);
            rx3.recv();
        });

        select! (
            a = rx1.recv() => { assert_eq!(a, 1); },
            a = rx2.recv() => { assert_eq!(a, 2); }
        )
        select! (
            a = rx1.recv() => { assert_eq!(a, 1); },
            a = rx2.recv() => { assert_eq!(a, 2); }
        )
        assert_eq!(rx1.try_recv(), Err(Empty));
        assert_eq!(rx2.try_recv(), Err(Empty));
        tx3.send(());
    })

    test!(fn stress() {
        static AMT: int = 10000;
        let (tx1, rx1) = channel::<int>();
        let (tx2, rx2) = channel::<int>();
        let (tx3, rx3) = channel::<()>();

        spawn(proc() {
            for i in range(0, AMT) {
                if i % 2 == 0 {
                    tx1.send(i);
                } else {
                    tx2.send(i);
                }
                rx3.recv();
            }
        });

        for i in range(0, AMT) {
            select! (
                i1 = rx1.recv() => { assert!(i % 2 == 0 && i == i1); },
                i2 = rx2.recv() => { assert!(i % 2 == 1 && i == i2); }
            )
            tx3.send(());
        }
    })

    test!(fn cloning() {
        let (tx1, rx1) = channel::<int>();
        let (_tx2, rx2) = channel::<int>();
        let (tx3, rx3) = channel::<()>();

        spawn(proc() {
            rx3.recv();
            tx1.clone();
            assert_eq!(rx3.try_recv(), Err(Empty));
            tx1.send(2);
            rx3.recv();
        });

        tx3.send(());
        select!(
            _i1 = rx1.recv() => {},
            _i2 = rx2.recv() => fail!()
        )
        tx3.send(());
    })

    test!(fn cloning2() {
        let (tx1, rx1) = channel::<int>();
        let (_tx2, rx2) = channel::<int>();
        let (tx3, rx3) = channel::<()>();

        spawn(proc() {
            rx3.recv();
            tx1.clone();
            assert_eq!(rx3.try_recv(), Err(Empty));
            tx1.send(2);
            rx3.recv();
        });

        tx3.send(());
        select!(
            _i1 = rx1.recv() => {},
            _i2 = rx2.recv() => fail!()
        )
        tx3.send(());
    })

    test!(fn cloning3() {
        let (tx1, rx1) = channel::<()>();
        let (tx2, rx2) = channel::<()>();
        let (tx3, rx3) = channel::<()>();
        spawn(proc() {
            let s = Select::new();
            let mut h1 = s.handle(&rx1);
            let mut h2 = s.handle(&rx2);
            unsafe { h2.add(); }
            unsafe { h1.add(); }
            assert_eq!(s.wait(), h2.id);
            tx3.send(());
        });

        for _ in range(0u, 1000) { task::deschedule(); }
        drop(tx1.clone());
        tx2.send(());
        rx3.recv();
    })

    test!(fn preflight1() {
        let (tx, rx) = channel();
        tx.send(());
        select!(
            () = rx.recv() => {}
        )
    })

    test!(fn preflight2() {
        let (tx, rx) = channel();
        tx.send(());
        tx.send(());
        select!(
            () = rx.recv() => {}
        )
    })

    test!(fn preflight3() {
        let (tx, rx) = channel();
        drop(tx.clone());
        tx.send(());
        select!(
            () = rx.recv() => {}
        )
    })

    test!(fn preflight4() {
        let (tx, rx) = channel();
        tx.send(());
        let s = Select::new();
        let mut h = s.handle(&rx);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    })

    test!(fn preflight5() {
        let (tx, rx) = channel();
        tx.send(());
        tx.send(());
        let s = Select::new();
        let mut h = s.handle(&rx);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    })

    test!(fn preflight6() {
        let (tx, rx) = channel();
        drop(tx.clone());
        tx.send(());
        let s = Select::new();
        let mut h = s.handle(&rx);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    })

    test!(fn preflight7() {
        let (tx, rx) = channel::<()>();
        drop(tx);
        let s = Select::new();
        let mut h = s.handle(&rx);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    })

    test!(fn preflight8() {
        let (tx, rx) = channel();
        tx.send(());
        drop(tx);
        rx.recv();
        let s = Select::new();
        let mut h = s.handle(&rx);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    })

    test!(fn preflight9() {
        let (tx, rx) = channel();
        drop(tx.clone());
        tx.send(());
        drop(tx);
        rx.recv();
        let s = Select::new();
        let mut h = s.handle(&rx);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    })

    test!(fn oneshot_data_waiting() {
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        spawn(proc() {
            select! {
                () = rx1.recv() => {}
            }
            tx2.send(());
        });

        for _ in range(0u, 100) { task::deschedule() }
        tx1.send(());
        rx2.recv();
    })

    test!(fn stream_data_waiting() {
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        tx1.send(());
        tx1.send(());
        rx1.recv();
        rx1.recv();
        spawn(proc() {
            select! {
                () = rx1.recv() => {}
            }
            tx2.send(());
        });

        for _ in range(0u, 100) { task::deschedule() }
        tx1.send(());
        rx2.recv();
    })

    test!(fn shared_data_waiting() {
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        drop(tx1.clone());
        tx1.send(());
        rx1.recv();
        spawn(proc() {
            select! {
                () = rx1.recv() => {}
            }
            tx2.send(());
        });

        for _ in range(0u, 100) { task::deschedule() }
        tx1.send(());
        rx2.recv();
    })

    test!(fn sync1() {
        let (tx, rx) = sync_channel::<int>(1);
        tx.send(1);
        select! {
            n = rx.recv() => { assert_eq!(n, 1); }
        }
    })

    test!(fn sync2() {
        let (tx, rx) = sync_channel::<int>(0);
        spawn(proc() {
            for _ in range(0u, 100) { task::deschedule() }
            tx.send(1);
        });
        select! {
            n = rx.recv() => { assert_eq!(n, 1); }
        }
    })

    test!(fn sync3() {
        let (tx1, rx1) = sync_channel::<int>(0);
        let (tx2, rx2): (Sender<int>, Receiver<int>) = channel();
        spawn(proc() { tx1.send(1); });
        spawn(proc() { tx2.send(2); });
        select! {
            n = rx1.recv() => {
                assert_eq!(n, 1);
                assert_eq!(rx2.recv(), 2);
            },
            n = rx2.recv() => {
                assert_eq!(n, 2);
                assert_eq!(rx1.recv(), 1);
            }
        }
    })
}
