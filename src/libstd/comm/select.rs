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
use cell::Cell;
use iter::Iterator;
use kinds::marker;
use kinds::Send;
use ops::Drop;
use option::{Some, None, Option};
use ptr::RawPtr;
use result::{Ok, Err, Result};
use rt::local::Local;
use rt::task::{Task, BlockedTask};
use super::Port;
use uint;

macro_rules! select {
    (
        $($name:pat = $port:ident.$meth:ident() => $code:expr),+
    ) => ({
        use std::comm::Select;
        let sel = Select::new();
        $( let mut $port = sel.handle(&$port); )+
        unsafe {
            $( $port.add(); )+
        }
        let ret = sel.wait();
        $( if ret == $port.id() { let $name = $port.$meth(); $code } else )+
        { unreachable!() }
    })
}

/// The "port set" of the select interface. This structure is used to manage a
/// set of ports which are being selected over.
pub struct Select {
    priv head: *mut Handle<'static, ()>,
    priv tail: *mut Handle<'static, ()>,
    priv next_id: Cell<uint>,
    priv marker1: marker::NoSend,
    priv marker2: marker::NoFreeze,
}

/// A handle to a port which is currently a member of a `Select` set of ports.
/// This handle is used to keep the port in the set as well as interact with the
/// underlying port.
pub struct Handle<'port, T> {
    /// The ID of this handle, used to compare against the return value of
    /// `Select::wait()`
    priv id: uint,
    priv selector: &'port Select,
    priv next: *mut Handle<'static, ()>,
    priv prev: *mut Handle<'static, ()>,
    priv added: bool,
    priv packet: &'port Packet,

    // due to our fun transmutes, we be sure to place this at the end. (nothing
    // previous relies on T)
    priv port: &'port Port<T>,
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
            marker2: marker::NoFreeze,
            head: 0 as *mut Handle<'static, ()>,
            tail: 0 as *mut Handle<'static, ()>,
            next_id: Cell::new(1),
        }
    }

    /// Creates a new handle into this port set for a new port. Note that this
    /// does *not* add the port to the port set, for that you must call the
    /// `add` method on the handle itself.
    pub fn handle<'a, T: Send>(&'a self, port: &'a Port<T>) -> Handle<'a, T> {
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        Handle {
            id: id,
            selector: self,
            next: 0 as *mut Handle<'static, ()>,
            prev: 0 as *mut Handle<'static, ()>,
            added: false,
            port: port,
            packet: port,
        }
    }

    /// Waits for an event on this port set. The returned value is *not* an
    /// index, but rather an id. This id can be queried against any active
    /// `Handle` structures (each one has an `id` method). The handle with
    /// the matching `id` will have some sort of event available on it. The
    /// event could either be that data is available or the corresponding
    /// channel has been closed.
    pub fn wait(&self) -> uint {
        self.wait2(false)
    }

    /// Helper method for skipping the preflight checks during testing
    fn wait2(&self, do_preflight_checks: bool) -> uint {
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
            // immediately so we can go unblock on all the other ports.
            let task: ~Task = Local::take();
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

impl<'port, T: Send> Handle<'port, T> {
    /// Retrieve the id of this handle.
    #[inline]
    pub fn id(&self) -> uint { self.id }

    /// Receive a value on the underlying port. Has the same semantics as
    /// `Port.recv`
    pub fn recv(&mut self) -> T { self.port.recv() }
    /// Block to receive a value on the underlying port, returning `Some` on
    /// success or `None` if the channel disconnects. This function has the same
    /// semantics as `Port.recv_opt`
    pub fn recv_opt(&mut self) -> Option<T> { self.port.recv_opt() }

    /// Adds this handle to the port set that the handle was created from. This
    /// method can be called multiple times, but it has no effect if `add` was
    /// called previously.
    ///
    /// This method is unsafe because it requires that the `Handle` is not moved
    /// while it is added to the `Select` set.
    pub unsafe fn add(&mut self) {
        if self.added { return }
        let selector: &mut Select = cast::transmute(&*self.selector);
        let me: *mut Handle<'static, ()> = cast::transmute(&*self);

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

        let selector: &mut Select = cast::transmute(&*self.selector);
        let me: *mut Handle<'static, ()> = cast::transmute(&*self);

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
impl<'port, T: Send> Drop for Handle<'port, T> {
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
    use super::super::*;
    use prelude::*;

    test!(fn smoke() {
        let (p1, c1) = Chan::<int>::new();
        let (p2, c2) = Chan::<int>::new();
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
            bar = p2.recv_opt() => { assert_eq!(bar, None); }
        )
    })

    test!(fn smoke2() {
        let (p1, _c1) = Chan::<int>::new();
        let (p2, _c2) = Chan::<int>::new();
        let (p3, _c3) = Chan::<int>::new();
        let (p4, _c4) = Chan::<int>::new();
        let (p5, c5) = Chan::<int>::new();
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
        let (p1, _c1) = Chan::<int>::new();
        let (p2, c2) = Chan::<int>::new();
        drop(c2);

        select! (
            _a1 = p1.recv_opt() => { fail!() },
            a2 = p2.recv_opt() => { assert_eq!(a2, None); }
        )
    })

    test!(fn unblocks() {
        let (p1, c1) = Chan::<int>::new();
        let (p2, _c2) = Chan::<int>::new();
        let (p3, c3) = Chan::<int>::new();

        spawn(proc() {
            for _ in range(0, 20) { task::deschedule(); }
            c1.send(1);
            p3.recv();
            for _ in range(0, 20) { task::deschedule(); }
        });

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
        let (p1, c1) = Chan::<int>::new();
        let (p2, c2) = Chan::<int>::new();
        let (p3, c3) = Chan::<()>::new();

        spawn(proc() {
            for _ in range(0, 20) { task::deschedule(); }
            c1.send(1);
            c2.send(2);
            p3.recv();
        });

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
        let (p1, c1) = Chan::<int>::new();
        let (p2, c2) = Chan::<int>::new();
        let (p3, c3) = Chan::<()>::new();

        spawn(proc() {
            for i in range(0, AMT) {
                if i % 2 == 0 {
                    c1.send(i);
                } else {
                    c2.send(i);
                }
                p3.recv();
            }
        });

        for i in range(0, AMT) {
            select! (
                i1 = p1.recv() => { assert!(i % 2 == 0 && i == i1); },
                i2 = p2.recv() => { assert!(i % 2 == 1 && i == i2); }
            )
            c3.send(());
        }
    })

    test!(fn cloning() {
        let (p1, c1) = Chan::<int>::new();
        let (p2, _c2) = Chan::<int>::new();
        let (p3, c3) = Chan::<()>::new();

        spawn(proc() {
            p3.recv();
            c1.clone();
            assert_eq!(p3.try_recv(), Empty);
            c1.send(2);
            p3.recv();
        });

        c3.send(());
        select!(
            _i1 = p1.recv() => {},
            _i2 = p2.recv() => fail!()
        )
        c3.send(());
    })

    test!(fn cloning2() {
        let (p1, c1) = Chan::<int>::new();
        let (p2, _c2) = Chan::<int>::new();
        let (p3, c3) = Chan::<()>::new();

        spawn(proc() {
            p3.recv();
            c1.clone();
            assert_eq!(p3.try_recv(), Empty);
            c1.send(2);
            p3.recv();
        });

        c3.send(());
        select!(
            _i1 = p1.recv() => {},
            _i2 = p2.recv() => fail!()
        )
        c3.send(());
    })

    test!(fn cloning3() {
        let (p1, c1) = Chan::<()>::new();
        let (p2, c2) = Chan::<()>::new();
        let (p, c) = Chan::new();
        spawn(proc() {
            let s = Select::new();
            let mut h1 = s.handle(&p1);
            let mut h2 = s.handle(&p2);
            unsafe { h2.add(); }
            unsafe { h1.add(); }
            assert_eq!(s.wait(), h2.id);
            c.send(());
        });

        for _ in range(0, 1000) { task::deschedule(); }
        drop(c1.clone());
        c2.send(());
        p.recv();
    })

    test!(fn preflight1() {
        let (p, c) = Chan::new();
        c.send(());
        select!(
            () = p.recv() => {}
        )
    })

    test!(fn preflight2() {
        let (p, c) = Chan::new();
        c.send(());
        c.send(());
        select!(
            () = p.recv() => {}
        )
    })

    test!(fn preflight3() {
        let (p, c) = Chan::new();
        drop(c.clone());
        c.send(());
        select!(
            () = p.recv() => {}
        )
    })

    test!(fn preflight4() {
        let (p, c) = Chan::new();
        c.send(());
        let s = Select::new();
        let mut h = s.handle(&p);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    })

    test!(fn preflight5() {
        let (p, c) = Chan::new();
        c.send(());
        c.send(());
        let s = Select::new();
        let mut h = s.handle(&p);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    })

    test!(fn preflight6() {
        let (p, c) = Chan::new();
        drop(c.clone());
        c.send(());
        let s = Select::new();
        let mut h = s.handle(&p);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    })

    test!(fn preflight7() {
        let (p, c) = Chan::<()>::new();
        drop(c);
        let s = Select::new();
        let mut h = s.handle(&p);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    })

    test!(fn preflight8() {
        let (p, c) = Chan::new();
        c.send(());
        drop(c);
        p.recv();
        let s = Select::new();
        let mut h = s.handle(&p);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    })

    test!(fn preflight9() {
        let (p, c) = Chan::new();
        drop(c.clone());
        c.send(());
        drop(c);
        p.recv();
        let s = Select::new();
        let mut h = s.handle(&p);
        unsafe { h.add(); }
        assert_eq!(s.wait2(false), h.id);
    })
}
