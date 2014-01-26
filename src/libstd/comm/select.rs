// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Selection over an array of ports
//!
//! This module contains the implementation machinery necessary for selecting
//! over a number of ports. One large goal of this module is to provide an
//! efficient interface to selecting over any port of any type.
//!
//! This is achieved through an architecture of a "port set" in which ports are
//! added to a set and then the entire set is waited on at once. The set can be
//! waited on multiple times to prevent re-adding each port to the set.
//!
//! Usage of this module is currently encouraged to go through the use of the
//! `select!` macro. This macro allows naturally binding of variables to the
//! received values of ports in a much more natural syntax then usage of the
//! `Select` structure directly.
//!
//! # Example
//!
//! ```rust,ignore
//! let (mut p1, c1) = Chan::new();
//! let (mut p2, c2) = Chan::new();
//!
//! c1.send(1);
//! c2.send(2);
//!
//! select! (
//!     val = p1.recv() => {
//!         assert_eq!(val, 1);
//!     }
//!     val = p2.recv() => {
//!         assert_eq!(val, 2);
//!     }
//! )
//! ```

#[allow(dead_code)];

use cast;
use comm;
use iter::Iterator;
use kinds::Send;
use ops::Drop;
use option::{Some, None, Option};
use ptr::RawPtr;
use result::{Ok, Err};
use rt::local::Local;
use rt::task::Task;
use super::{Packet, Port};
use sync::atomics::{Relaxed, SeqCst};
use task;
use uint;

macro_rules! select {
    (
        $name1:pat = $port1:ident.$meth1:ident() => $code1:expr,
        $($name:pat = $port:ident.$meth:ident() => $code:expr),*
    ) => ({
        use std::comm::Select;
        let sel = Select::new();
        let mut $port1 = sel.add(&mut $port1);
        $( let mut $port = sel.add(&mut $port); )*
        let ret = sel.wait();
        if ret == $port1.id { let $name1 = $port1.$meth1(); $code1 }
        $( else if ret == $port.id { let $name = $port.$meth(); $code } )*
        else { unreachable!() }
    })
}

/// The "port set" of the select interface. This structure is used to manage a
/// set of ports which are being selected over.
#[no_freeze]
#[no_send]
pub struct Select {
    priv head: *mut Packet,
    priv tail: *mut Packet,
    priv next_id: uint,
}

/// A handle to a port which is currently a member of a `Select` set of ports.
/// This handle is used to keep the port in the set as well as interact with the
/// underlying port.
pub struct Handle<'port, T> {
    /// A unique ID for this Handle.
    id: uint,
    priv selector: &'port Select,
    priv port: &'port mut Port<T>,
}

struct Packets { priv cur: *mut Packet }

impl Select {
    /// Creates a new selection structure. This set is initially empty and
    /// `wait` will fail!() if called.
    ///
    /// Usage of this struct directly can sometimes be burdensome, and usage is
    /// rather much easier through the `select!` macro.
    pub fn new() -> Select {
        Select {
            head: 0 as *mut Packet,
            tail: 0 as *mut Packet,
            next_id: 1,
        }
    }

    /// Adds a new port to this set, returning a handle which is then used to
    /// receive on the port.
    ///
    /// Note that this port parameter takes `&mut Port` instead of `&Port`. None
    /// of the methods of receiving on a port require `&mut self`, but `&mut` is
    /// used here in order to have the compiler guarantee that the same port is
    /// not added to this set more than once.
    ///
    /// When the returned handle falls out of scope, the port will be removed
    /// from this set. While the handle is in this set, usage of the port can be
    /// done through the `Handle`'s receiving methods.
    pub fn add<'a, T: Send>(&'a self, port: &'a mut Port<T>) -> Handle<'a, T> {
        let this = unsafe { cast::transmute_mut(self) };
        let id = this.next_id;
        this.next_id += 1;
        unsafe {
            let packet = port.queue.packet();
            assert!(!(*packet).selecting.load(Relaxed));
            assert_eq!((*packet).selection_id, 0);
            (*packet).selection_id = id;
            if this.head.is_null() {
                this.head = packet;
                this.tail = packet;
            } else {
                (*packet).select_prev = this.tail;
                assert!((*packet).select_next.is_null());
                (*this.tail).select_next = packet;
                this.tail = packet;
            }
        }
        Handle { id: id, selector: this, port: port }
    }

    /// Waits for an event on this port set. The returned valus is *not* and
    /// index, but rather an id. This id can be queried against any active
    /// `Handle` structures (each one has a public `id` field). The handle with
    /// the matching `id` will have some sort of event available on it. The
    /// event could either be that data is available or the corresponding
    /// channel has been closed.
    pub fn wait(&self) -> uint {
        // Note that this is currently an inefficient implementation. We in
        // theory have knowledge about all ports in the set ahead of time, so
        // this method shouldn't really have to iterate over all of them yet
        // again. The idea with this "port set" interface is to get the
        // interface right this time around, and later this implementation can
        // be optimized.
        //
        // This implementation can be summarized by:
        //
        //      fn select(ports) {
        //          if any port ready { return ready index }
        //          deschedule {
        //              block on all ports
        //          }
        //          unblock on all ports
        //          return ready index
        //      }
        //
        // Most notably, the iterations over all of the ports shouldn't be
        // necessary.
        unsafe {
            let mut amt = 0;
            for p in self.iter() {
                assert!(!(*p).selecting.load(Relaxed));
                amt += 1;
                if (*p).can_recv() {
                    return (*p).selection_id;
                }
            }
            assert!(amt > 0);

            let mut ready_index = amt;
            let mut ready_id = uint::MAX;
            let mut iter = self.iter().enumerate();

            // Acquire a number of blocking contexts, and block on each one
            // sequentially until one fails. If one fails, then abort
            // immediately so we can go unblock on all the other ports.
            let task: ~Task = Local::take();
            task.deschedule(amt, |task| {
                // Prepare for the block
                let (i, packet) = iter.next().unwrap();
                assert!((*packet).to_wake.is_none());
                (*packet).to_wake = Some(task);
                (*packet).selecting.store(true, SeqCst);

                if (*packet).decrement() {
                    Ok(())
                } else {
                    // Empty to_wake first to avoid tripping an assertion in
                    // abort_selection in the disconnected case.
                    let task = (*packet).to_wake.take_unwrap();
                    (*packet).abort_selection(false);
                    (*packet).selecting.store(false, SeqCst);
                    ready_index = i;
                    ready_id = (*packet).selection_id;
                    Err(task)
                }
            });

            // Abort the selection process on each port. If the abort process
            // returns `true`, then that means that the port is ready to receive
            // some data. Note that this also means that the port may have yet
            // to have fully read the `to_wake` field and woken us up (although
            // the wakeup is guaranteed to fail).
            //
            // This situation happens in the window of where a sender invokes
            // increment(), sees -1, and then decides to wake up the task. After
            // all this is done, the sending thread will set `selecting` to
            // `false`. Until this is done, we cannot return. If we were to
            // return, then a sender could wake up a port which has gone back to
            // sleep after this call to `select`.
            //
            // Note that it is a "fairly small window" in which an increment()
            // views that it should wake a thread up until the `selecting` bit
            // is set to false. For now, the implementation currently just spins
            // in a yield loop. This is very distasteful, but this
            // implementation is already nowhere near what it should ideally be.
            // A rewrite should focus on avoiding a yield loop, and for now this
            // implementation is tying us over to a more efficient "don't
            // iterate over everything every time" implementation.
            for packet in self.iter().take(ready_index) {
                if (*packet).abort_selection(true) {
                    ready_id = (*packet).selection_id;
                    while (*packet).selecting.load(Relaxed) {
                        task::deschedule();
                    }
                }
            }

            // Sanity check for now to make sure that everyone is turned off.
            for packet in self.iter() {
                assert!(!(*packet).selecting.load(Relaxed));
            }

            assert!(ready_id != uint::MAX);
            return ready_id;
        }
    }

    unsafe fn remove(&self, packet: *mut Packet) {
        let this = cast::transmute_mut(self);
        assert!(!(*packet).selecting.load(Relaxed));
        if (*packet).select_prev.is_null() {
            assert_eq!(packet, this.head);
            this.head = (*packet).select_next;
        } else {
            (*(*packet).select_prev).select_next = (*packet).select_next;
        }
        if (*packet).select_next.is_null() {
            assert_eq!(packet, this.tail);
            this.tail = (*packet).select_prev;
        } else {
            (*(*packet).select_next).select_prev = (*packet).select_prev;
        }
        (*packet).select_next = 0 as *mut Packet;
        (*packet).select_prev = 0 as *mut Packet;
        (*packet).selection_id = 0;
    }

    fn iter(&self) -> Packets { Packets { cur: self.head } }
}

impl<'port, T: Send> Handle<'port, T> {
    /// Receive a value on the underlying port. Has the same semantics as
    /// `Port.recv`
    pub fn recv(&mut self) -> T { self.port.recv() }
    /// Block to receive a value on the underlying port, returning `Some` on
    /// success or `None` if the channel disconnects. This function has the same
    /// semantics as `Port.recv_opt`
    pub fn recv_opt(&mut self) -> Option<T> { self.port.recv_opt() }
    /// Immediately attempt to receive a value on a port, this function will
    /// never block. Has the same semantics as `Port.try_recv`.
    pub fn try_recv(&mut self) -> comm::TryRecvResult<T> {
        self.port.try_recv()
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
impl<'port, T: Send> Drop for Handle<'port, T> {
    fn drop(&mut self) {
        unsafe { self.selector.remove(self.port.queue.packet()) }
    }
}

impl Iterator<*mut Packet> for Packets {
    fn next(&mut self) -> Option<*mut Packet> {
        if self.cur.is_null() {
            None
        } else {
            let ret = Some(self.cur);
            unsafe { self.cur = (*self.cur).select_next; }
            ret
        }
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod test {
    use super::super::*;
    use prelude::*;

    test!(fn smoke() {
        let (mut p1, c1) = Chan::<int>::new();
        let (mut p2, c2) = Chan::<int>::new();
        c1.send(1);
        select! (
            foo = p1.recv() => { assert_eq!(foo, 1); },
            _bar = p2.recv() => { fail!() }
        )
        c2.send(2);
        select! (
            _foo = p1.recv() => { fail!() },
            bar = p2.recv() => { assert_eq!(bar, 2) }
        )
        drop(c1);
        select! (
            foo = p1.recv_opt() => { assert_eq!(foo, None); },
            _bar = p2.recv() => { fail!() }
        )
        drop(c2);
        select! (
            bar = p2.recv_opt() => { assert_eq!(bar, None); },
        )
    })

    test!(fn smoke2() {
        let (mut p1, _c1) = Chan::<int>::new();
        let (mut p2, _c2) = Chan::<int>::new();
        let (mut p3, _c3) = Chan::<int>::new();
        let (mut p4, _c4) = Chan::<int>::new();
        let (mut p5, c5) = Chan::<int>::new();
        c5.send(4);
        select! (
            _foo = p1.recv() => { fail!("1") },
            _foo = p2.recv() => { fail!("2") },
            _foo = p3.recv() => { fail!("3") },
            _foo = p4.recv() => { fail!("4") },
            foo = p5.recv() => { assert_eq!(foo, 4); }
        )
    })

    test!(fn closed() {
        let (mut p1, _c1) = Chan::<int>::new();
        let (mut p2, c2) = Chan::<int>::new();
        drop(c2);

        select! (
            _a1 = p1.recv_opt() => { fail!() },
            a2 = p2.recv_opt() => { assert_eq!(a2, None); }
        )
    })

    test!(fn unblocks() {
        let (mut p1, c1) = Chan::<int>::new();
        let (mut p2, _c2) = Chan::<int>::new();
        let (p3, c3) = Chan::<int>::new();

        do spawn {
            20.times(task::deschedule);
            c1.send(1);
            p3.recv();
            20.times(task::deschedule);
        }

        select! (
            a = p1.recv() => { assert_eq!(a, 1); },
            _b = p2.recv() => { fail!() }
        )
        c3.send(1);
        select! (
            a = p1.recv_opt() => { assert_eq!(a, None); },
            _b = p2.recv() => { fail!() }
        )
    })

    test!(fn both_ready() {
        let (mut p1, c1) = Chan::<int>::new();
        let (mut p2, c2) = Chan::<int>::new();
        let (p3, c3) = Chan::<()>::new();

        do spawn {
            20.times(task::deschedule);
            c1.send(1);
            c2.send(2);
            p3.recv();
        }

        select! (
            a = p1.recv() => { assert_eq!(a, 1); },
            a = p2.recv() => { assert_eq!(a, 2); }
        )
        select! (
            a = p1.recv() => { assert_eq!(a, 1); },
            a = p2.recv() => { assert_eq!(a, 2); }
        )
        assert_eq!(p1.try_recv(), Empty);
        assert_eq!(p2.try_recv(), Empty);
        c3.send(());
    })

    test!(fn stress() {
        static AMT: int = 10000;
        let (mut p1, c1) = Chan::<int>::new();
        let (mut p2, c2) = Chan::<int>::new();
        let (p3, c3) = Chan::<()>::new();

        do spawn {
            for i in range(0, AMT) {
                if i % 2 == 0 {
                    c1.send(i);
                } else {
                    c2.send(i);
                }
                p3.recv();
            }
        }

        for i in range(0, AMT) {
            select! (
                i1 = p1.recv() => { assert!(i % 2 == 0 && i == i1); },
                i2 = p2.recv() => { assert!(i % 2 == 1 && i == i2); }
            )
            c3.send(());
        }
    })
}
