// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Communication primitives for concurrent tasks (`Chan` and `Port` types)
//!
//! Rust makes it very difficult to share data among tasks to prevent race
//! conditions and to improve parallelism, but there is often a need for
//! communication between concurrent tasks. The primitives defined in this
//! module are the building blocks for synchronization in rust.
//!
//! This module currently provides two types:
//!
//! * `Chan`
//! * `Port`
//!
//! `Chan` is used to send data to a `Port`. A `Chan` is clone-able such that
//! many tasks can send simultaneously to one receiving port. These
//! communication primitives are *task blocking*, not *thread blocking*. This
//! means that if one task is blocked on a channel, other tasks can continue to
//! make progress.
//!
//! Rust channels can be used as if they have an infinite internal buffer. What
//! this means is that the `send` operation will never block. `Port`s, on the
//! other hand, will block the task if there is no data to be received.
//!
//! ## Failure Propagation
//!
//! In addition to being a core primitive for communicating in rust, channels
//! and ports are the points at which failure is propagated among tasks.
//! Whenever the one half of channel is closed, the other half will have its
//! next operation `fail!`. The purpose of this is to allow propagation of
//! failure among tasks that are linked to one another via channels.
//!
//! There are methods on both of `Chan` and `Port` to perform their respective
//! operations without failing, however.
//!
//! ## Outside the Runtime
//!
//! All channels and ports work seamlessly inside and outside of the rust
//! runtime. This means that code may use channels to communicate information
//! inside and outside of the runtime. For example, if rust were embedded as an
//! FFI module in another application, the rust runtime would probably be
//! running in its own external thread pool. Channels created can communicate
//! from the native application threads to the rust threads through the use of
//! native mutexes and condition variables.
//!
//! What this means is that if a native thread is using a channel, execution
//! will be blocked accordingly by blocking the OS thread.
//!
//! # Example
//!
//! ```rust,should_fail
//! // Create a simple streaming channel
//! let (port, chan) = Chan::new();
//! spawn(proc() {
//!     chan.send(10);
//! })
//! assert_eq!(port.recv(), 10);
//!
//! // Create a shared channel which can be sent along from many tasks
//! let (port, chan) = Chan::new();
//! for i in range(0, 10) {
//!     let chan = chan.clone();
//!     spawn(proc() {
//!         chan.send(i);
//!     })
//! }
//!
//! for _ in range(0, 10) {
//!     let j = port.recv();
//!     assert!(0 <= j && j < 10);
//! }
//!
//! // The call to recv() will fail!() because the channel has already hung
//! // up (or been deallocated)
//! let (port, chan) = Chan::<int>::new();
//! drop(chan);
//! port.recv();
//! ```

// A description of how Rust's channel implementation works
//
// Channels are supposed to be the basic building block for all other
// concurrent primitives that are used in Rust. As a result, the channel type
// needs to be highly optimized, flexible, and broad enough for use everywhere.
//
// The choice of implementation of all channels is to be built on lock-free data
// structures. The channels themselves are then consequently also lock-free data
// structures. As always with lock-free code, this is a very "here be dragons"
// territory, especially because I'm unaware of any academic papers which have
// gone into great length about channels of these flavors.
//
// ## Flavors of channels
//
// From the perspective of a consumer of this library, there is only one flavor
// of channel. This channel can be used as a stream and cloned to allow multiple
// senders. Under the hood, however, there are actually three flavors of
// channels in play.
//
// * Oneshots - these channels are highly optimized for the one-send use case.
//              They contain as few atomics as possible and involve one and
//              exactly one allocation.
// * Streams - these channels are optimized for the non-shared use case. They
//             use a different concurrent queue which is more tailored for this
//             use case. The initial allocation of this flavor of channel is not
//             optimized.
// * Shared - this is the most general form of channel that this module offers,
//            a channel with multiple senders. This type is as optimized as it
//            can be, but the previous two types mentioned are much faster for
//            their use-cases.
//
// ## Concurrent queues
//
// The basic idea of Rust's Chan/Port types is that send() never blocks, but
// recv() obviously blocks. This means that under the hood there must be some
// shared and concurrent queue holding all of the actual data.
//
// With two flavors of channels, two flavors of queues are also used. We have
// chosen to use queues from a well-known author which are abbreviated as SPSC
// and MPSC (single producer, single consumer and multiple producer, single
// consumer). SPSC queues are used for streams while MPSC queues are used for
// shared channels.
//
// ### SPSC optimizations
//
// The SPSC queue found online is essentially a linked list of nodes where one
// half of the nodes are the "queue of data" and the other half of nodes are a
// cache of unused nodes. The unused nodes are used such that an allocation is
// not required on every push() and a free doesn't need to happen on every
// pop().
//
// As found online, however, the cache of nodes is of an infinite size. This
// means that if a channel at one point in its life had 50k items in the queue,
// then the queue will always have the capacity for 50k items. I believed that
// this was an unnecessary limitation of the implementation, so I have altered
// the queue to optionally have a bound on the cache size.
//
// By default, streams will have an unbounded SPSC queue with a small-ish cache
// size. The hope is that the cache is still large enough to have very fast
// send() operations while not too large such that millions of channels can
// coexist at once.
//
// ### MPSC optimizations
//
// Right now the MPSC queue has not been optimized. Like the SPSC queue, it uses
// a linked list under the hood to earn its unboundedness, but I have not put
// forth much effort into having a cache of nodes similar to the SPSC queue.
//
// For now, I believe that this is "ok" because shared channels are not the most
// common type, but soon we may wish to revisit this queue choice and determine
// another candidate for backend storage of shared channels.
//
// ## Overview of the Implementation
//
// Now that there's a little background on the concurrent queues used, it's
// worth going into much more detail about the channels themselves. The basic
// pseudocode for a send/recv are:
//
//
//      send(t)                             recv()
//        queue.push(t)                       return if queue.pop()
//        if increment() == -1                deschedule {
//          wakeup()                            if decrement() > 0
//                                                cancel_deschedule()
//                                            }
//                                            queue.pop()
//
// As mentioned before, there are no locks in this implementation, only atomic
// instructions are used.
//
// ### The internal atomic counter
//
// Every channel/port/shared channel have a shared counter with their
// counterparts to keep track of the size of the queue. This counter is used to
// abort descheduling by the receiver and to know when to wake up on the sending
// side.
//
// As seen in the pseudocode, senders will increment this count and receivers
// will decrement the count. The theory behind this is that if a sender sees a
// -1 count, it will wake up the receiver, and if the receiver sees a 1+ count,
// then it doesn't need to block.
//
// The recv() method has a beginning call to pop(), and if successful, it needs
// to decrement the count. It is a crucial implementation detail that this
// decrement does *not* happen to the shared counter. If this were the case,
// then it would be possible for the counter to be very negative when there were
// no receivers waiting, in which case the senders would have to determine when
// it was actually appropriate to wake up a receiver.
//
// Instead, the "steal count" is kept track of separately (not atomically
// because it's only used by ports), and then the decrement() call when
// descheduling will lump in all of the recent steals into one large decrement.
//
// The implication of this is that if a sender sees a -1 count, then there's
// guaranteed to be a waiter waiting!
//
// ## Native Implementation
//
// A major goal of these channels is to work seamlessly on and off the runtime.
// All of the previous race conditions have been worded in terms of
// scheduler-isms (which is obviously not available without the runtime).
//
// For now, native usage of channels (off the runtime) will fall back onto
// mutexes/cond vars for descheduling/atomic decisions. The no-contention path
// is still entirely lock-free, the "deschedule" blocks above are surrounded by
// a mutex and the "wakeup" blocks involve grabbing a mutex and signaling on a
// condition variable.
//
// ## Select
//
// Being able to support selection over channels has greatly influenced this
// design, and not only does selection need to work inside the runtime, but also
// outside the runtime.
//
// The implementation is fairly straightforward. The goal of select() is not to
// return some data, but only to return which channel can receive data without
// blocking. The implementation is essentially the entire blocking procedure
// followed by an increment as soon as its woken up. The cancellation procedure
// involves an increment and swapping out of to_wake to acquire ownership of the
// task to unblock.
//
// Sadly this current implementation requires multiple allocations, so I have
// seen the throughput of select() be much worse than it should be. I do not
// believe that there is anything fundamental which needs to change about these
// channels, however, in order to support a more efficient select().
//
// # Conclusion
//
// And now that you've seen all the races that I found and attempted to fix,
// here's the code for you to find some more!

use cast;
use cell::Cell;
use clone::Clone;
use iter::Iterator;
use kinds::Send;
use kinds::marker;
use mem;
use ops::Drop;
use option::{Some, None, Option};
use result::{Ok, Err, Result};
use rt::local::Local;
use rt::task::{Task, BlockedTask};
use sync::arc::UnsafeArc;

pub use comm::select::{Select, Handle};

macro_rules! test (
    { fn $name:ident() $b:block $($a:attr)*} => (
        mod $name {
            #[allow(unused_imports)];

            use native;
            use comm::*;
            use prelude::*;
            use super::*;
            use super::super::*;
            use task;

            fn f() $b

            $($a)* #[test] fn uv() { f() }
            $($a)* #[test] fn native() {
                use native;
                let (p, c) = Chan::new();
                native::task::spawn(proc() { c.send(f()) });
                p.recv();
            }
        }
    )
)

mod select;
mod oneshot;
mod stream;
mod shared;

// Use a power of 2 to allow LLVM to optimize to something that's not a
// division, this is hit pretty regularly.
static RESCHED_FREQ: int = 256;

/// The receiving-half of Rust's channel type. This half can only be owned by
/// one task
pub struct Port<T> {
    priv inner: Flavor<T>,
    priv receives: Cell<uint>,
    // can't share in an arc
    priv marker: marker::NoFreeze,
}

/// An iterator over messages received on a port, this iterator will block
/// whenever `next` is called, waiting for a new message, and `None` will be
/// returned when the corresponding channel has hung up.
pub struct Messages<'a, T> {
    priv port: &'a Port<T>
}

/// The sending-half of Rust's channel type. This half can only be owned by one
/// task
pub struct Chan<T> {
    priv inner: Flavor<T>,
    priv sends: Cell<uint>,
    // can't share in an arc
    priv marker: marker::NoFreeze,
}

/// This enumeration is the list of the possible reasons that try_recv could not
/// return data when called.
#[deriving(Eq, Clone)]
pub enum TryRecvResult<T> {
    /// This channel is currently empty, but the sender(s) have not yet
    /// disconnected, so data may yet become available.
    Empty,
    /// This channel's sending half has become disconnected, and there will
    /// never be any more data received on this channel
    Disconnected,
    /// The channel had some data and we successfully popped it
    Data(T),
}

enum Flavor<T> {
    Oneshot(UnsafeArc<oneshot::Packet<T>>),
    Stream(UnsafeArc<stream::Packet<T>>),
    Shared(UnsafeArc<shared::Packet<T>>),
}

impl<T: Send> Chan<T> {
    /// Creates a new port/channel pair. All data send on the channel returned
    /// will become available on the port as well. See the documentation of
    /// `Port` and `Chan` to see what's possible with them.
    pub fn new() -> (Port<T>, Chan<T>) {
        let (a, b) = UnsafeArc::new2(oneshot::Packet::new());
        (Port::my_new(Oneshot(a)), Chan::my_new(Oneshot(b)))
    }

    fn my_new(inner: Flavor<T>) -> Chan<T> {
        Chan { inner: inner, sends: Cell::new(0), marker: marker::NoFreeze }
    }

    /// Sends a value along this channel to be received by the corresponding
    /// port.
    ///
    /// Rust channels are infinitely buffered so this method will never block.
    ///
    /// # Failure
    ///
    /// This function will fail if the other end of the channel has hung up.
    /// This means that if the corresponding port has fallen out of scope, this
    /// function will trigger a fail message saying that a message is being sent
    /// on a closed channel.
    ///
    /// Note that if this function does *not* fail, it does not mean that the
    /// data will be successfully received. All sends are placed into a queue,
    /// so it is possible for a send to succeed (the other end is alive), but
    /// then the other end could immediately disconnect.
    ///
    /// The purpose of this functionality is to propagate failure among tasks.
    /// If failure is not desired, then consider using the `try_send` method
    pub fn send(&self, t: T) {
        if !self.try_send(t) {
            fail!("sending on a closed channel");
        }
    }

    /// Attempts to send a value on this channel, returning whether it was
    /// successfully sent.
    ///
    /// A successful send occurs when it is determined that the other end of the
    /// channel has not hung up already. An unsuccessful send would be one where
    /// the corresponding port has already been deallocated. Note that a return
    /// value of `false` means that the data will never be received, but a
    /// return value of `true` does *not* mean that the data will be received.
    /// It is possible for the corresponding port to hang up immediately after
    /// this function returns `true`.
    ///
    /// Like `send`, this method will never block. If the failure of send cannot
    /// be tolerated, then this method should be used instead.
    pub fn try_send(&self, t: T) -> bool {
        // In order to prevent starvation of other tasks in situations where
        // a task sends repeatedly without ever receiving, we occassionally
        // yield instead of doing a send immediately.  Only doing this if
        // we're doing a rescheduling send, otherwise the caller is
        // expecting not to context switch.
        //
        // Note that we don't unconditionally attempt to yield because the
        // TLS overhead can be a bit much.
        let cnt = self.sends.get() + 1;
        self.sends.set(cnt);
        if cnt % (RESCHED_FREQ as uint) == 0 {
            let task: ~Task = Local::take();
            task.maybe_yield();
        }

        let (new_inner, ret) = match self.inner {
            Oneshot(ref p) => {
                let p = p.get();
                unsafe {
                    if !(*p).sent() {
                        return (*p).send(t);
                    } else {
                        let (a, b) = UnsafeArc::new2(stream::Packet::new());
                        match (*p).upgrade(Port::my_new(Stream(b))) {
                            oneshot::UpSuccess => {
                                (*a.get()).send(t);
                                (a, true)
                            }
                            oneshot::UpDisconnected => (a, false),
                            oneshot::UpWoke(task) => {
                                (*a.get()).send(t);
                                task.wake().map(|t| t.reawaken());
                                (a, true)
                            }
                        }
                    }
                }
            }
            Stream(ref p) => return unsafe { (*p.get()).send(t) },
            Shared(ref p) => return unsafe { (*p.get()).send(t) },
        };

        unsafe {
            let mut tmp = Chan::my_new(Stream(new_inner));
            mem::swap(&mut cast::transmute_mut(self).inner, &mut tmp.inner);
        }
        return ret;
    }
}

impl<T: Send> Clone for Chan<T> {
    fn clone(&self) -> Chan<T> {
        let (packet, sleeper) = match self.inner {
            Oneshot(ref p) => {
                let (a, b) = UnsafeArc::new2(shared::Packet::new());
                match unsafe { (*p.get()).upgrade(Port::my_new(Shared(a))) } {
                    oneshot::UpSuccess | oneshot::UpDisconnected => (b, None),
                    oneshot::UpWoke(task) => (b, Some(task))
                }
            }
            Stream(ref p) => {
                let (a, b) = UnsafeArc::new2(shared::Packet::new());
                match unsafe { (*p.get()).upgrade(Port::my_new(Shared(a))) } {
                    stream::UpSuccess | stream::UpDisconnected => (b, None),
                    stream::UpWoke(task) => (b, Some(task)),
                }
            }
            Shared(ref p) => {
                unsafe { (*p.get()).clone_chan(); }
                return Chan::my_new(Shared(p.clone()));
            }
        };

        unsafe {
            (*packet.get()).inherit_blocker(sleeper);

            let mut tmp = Chan::my_new(Shared(packet.clone()));
            mem::swap(&mut cast::transmute_mut(self).inner, &mut tmp.inner);
        }
        Chan::my_new(Shared(packet))
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for Chan<T> {
    fn drop(&mut self) {
        match self.inner {
            Oneshot(ref mut p) => unsafe { (*p.get()).drop_chan(); },
            Stream(ref mut p) => unsafe { (*p.get()).drop_chan(); },
            Shared(ref mut p) => unsafe { (*p.get()).drop_chan(); },
        }
    }
}

impl<T: Send> Port<T> {
    fn my_new(inner: Flavor<T>) -> Port<T> {
        Port { inner: inner, receives: Cell::new(0), marker: marker::NoFreeze }
    }

    /// Blocks waiting for a value on this port
    ///
    /// This function will block if necessary to wait for a corresponding send
    /// on the channel from its paired `Chan` structure. This port will be woken
    /// up when data is ready, and the data will be returned.
    ///
    /// # Failure
    ///
    /// Similar to channels, this method will trigger a task failure if the
    /// other end of the channel has hung up (been deallocated). The purpose of
    /// this is to propagate failure among tasks.
    ///
    /// If failure is not desired, then there are two options:
    ///
    /// * If blocking is still desired, the `recv_opt` method will return `None`
    ///   when the other end hangs up
    ///
    /// * If blocking is not desired, then the `try_recv` method will attempt to
    ///   peek at a value on this port.
    pub fn recv(&self) -> T {
        match self.recv_opt() {
            Some(t) => t,
            None => fail!("receiving on a closed channel"),
        }
    }

    /// Attempts to return a pending value on this port without blocking
    ///
    /// This method will never block the caller in order to wait for data to
    /// become available. Instead, this will always return immediately with a
    /// possible option of pending data on the channel.
    ///
    /// This is useful for a flavor of "optimistic check" before deciding to
    /// block on a port.
    ///
    /// This function cannot fail.
    pub fn try_recv(&self) -> TryRecvResult<T> {
        // If a thread is spinning in try_recv, we should take the opportunity
        // to reschedule things occasionally. See notes above in scheduling on
        // sends for why this doesn't always hit TLS.
        let cnt = self.receives.get() + 1;
        self.receives.set(cnt);
        if cnt % (RESCHED_FREQ as uint) == 0 {
            let task: ~Task = Local::take();
            task.maybe_yield();
        }

        loop {
            let mut new_port = match self.inner {
                Oneshot(ref p) => {
                    match unsafe { (*p.get()).try_recv() } {
                        Ok(t) => return Data(t),
                        Err(oneshot::Empty) => return Empty,
                        Err(oneshot::Disconnected) => return Disconnected,
                        Err(oneshot::Upgraded(port)) => port,
                    }
                }
                Stream(ref p) => {
                    match unsafe { (*p.get()).try_recv() } {
                        Ok(t) => return Data(t),
                        Err(stream::Empty) => return Empty,
                        Err(stream::Disconnected) => return Disconnected,
                        Err(stream::Upgraded(port)) => port,
                    }
                }
                Shared(ref p) => {
                    match unsafe { (*p.get()).try_recv() } {
                        Ok(t) => return Data(t),
                        Err(shared::Empty) => return Empty,
                        Err(shared::Disconnected) => return Disconnected,
                    }
                }
            };
            unsafe {
                mem::swap(&mut cast::transmute_mut(self).inner,
                          &mut new_port.inner);
            }
        }
    }

    /// Attempt to wait for a value on this port, but does not fail if the
    /// corresponding channel has hung up.
    ///
    /// This implementation of iterators for ports will always block if there is
    /// not data available on the port, but it will not fail in the case that
    /// the channel has been deallocated.
    ///
    /// In other words, this function has the same semantics as the `recv`
    /// method except for the failure aspect.
    ///
    /// If the channel has hung up, then `None` is returned. Otherwise `Some` of
    /// the value found on the port is returned.
    pub fn recv_opt(&self) -> Option<T> {
        loop {
            let mut new_port = match self.inner {
                Oneshot(ref p) => {
                    match unsafe { (*p.get()).recv() } {
                        Ok(t) => return Some(t),
                        Err(oneshot::Empty) => return unreachable!(),
                        Err(oneshot::Disconnected) => return None,
                        Err(oneshot::Upgraded(port)) => port,
                    }
                }
                Stream(ref p) => {
                    match unsafe { (*p.get()).recv() } {
                        Ok(t) => return Some(t),
                        Err(stream::Empty) => return unreachable!(),
                        Err(stream::Disconnected) => return None,
                        Err(stream::Upgraded(port)) => port,
                    }
                }
                Shared(ref p) => {
                    match unsafe { (*p.get()).recv() } {
                        Ok(t) => return Some(t),
                        Err(shared::Empty) => return unreachable!(),
                        Err(shared::Disconnected) => return None,
                    }
                }
            };
            unsafe {
                mem::swap(&mut cast::transmute_mut(self).inner,
                          &mut new_port.inner);
            }
        }
    }

    /// Returns an iterator which will block waiting for messages, but never
    /// `fail!`. It will return `None` when the channel has hung up.
    pub fn iter<'a>(&'a self) -> Messages<'a, T> {
        Messages { port: self }
    }
}

impl<T: Send> select::Packet for Port<T> {
    fn can_recv(&self) -> bool {
        loop {
            let mut new_port = match self.inner {
                Oneshot(ref p) => {
                    match unsafe { (*p.get()).can_recv() } {
                        Ok(ret) => return ret,
                        Err(upgrade) => upgrade,
                    }
                }
                Stream(ref p) => {
                    match unsafe { (*p.get()).can_recv() } {
                        Ok(ret) => return ret,
                        Err(upgrade) => upgrade,
                    }
                }
                Shared(ref p) => {
                    return unsafe { (*p.get()).can_recv() };
                }
            };
            unsafe {
                mem::swap(&mut cast::transmute_mut(self).inner,
                          &mut new_port.inner);
            }
        }
    }

    fn start_selection(&self, mut task: BlockedTask) -> Result<(), BlockedTask>{
        loop {
            let (t, mut new_port) = match self.inner {
                Oneshot(ref p) => {
                    match unsafe { (*p.get()).start_selection(task) } {
                        oneshot::SelSuccess => return Ok(()),
                        oneshot::SelCanceled(task) => return Err(task),
                        oneshot::SelUpgraded(t, port) => (t, port),
                    }
                }
                Stream(ref p) => {
                    match unsafe { (*p.get()).start_selection(task) } {
                        stream::SelSuccess => return Ok(()),
                        stream::SelCanceled(task) => return Err(task),
                        stream::SelUpgraded(t, port) => (t, port),
                    }
                }
                Shared(ref p) => {
                    return unsafe { (*p.get()).start_selection(task) };
                }
            };
            task = t;
            unsafe {
                mem::swap(&mut cast::transmute_mut(self).inner,
                          &mut new_port.inner);
            }
        }
    }

    fn abort_selection(&self) -> bool {
        let mut was_upgrade = false;
        loop {
            let result = match self.inner {
                Oneshot(ref p) => unsafe { (*p.get()).abort_selection() },
                Stream(ref p) => unsafe {
                    (*p.get()).abort_selection(was_upgrade)
                },
                Shared(ref p) => return unsafe {
                    (*p.get()).abort_selection(was_upgrade)
                },
            };
            let mut new_port = match result { Ok(b) => return b, Err(p) => p };
            was_upgrade = true;
            unsafe {
                mem::swap(&mut cast::transmute_mut(self).inner,
                          &mut new_port.inner);
            }
        }
    }
}

impl<'a, T: Send> Iterator<T> for Messages<'a, T> {
    fn next(&mut self) -> Option<T> { self.port.recv_opt() }
}

#[unsafe_destructor]
impl<T: Send> Drop for Port<T> {
    fn drop(&mut self) {
        match self.inner {
            Oneshot(ref mut p) => unsafe { (*p.get()).drop_port(); },
            Stream(ref mut p) => unsafe { (*p.get()).drop_port(); },
            Shared(ref mut p) => unsafe { (*p.get()).drop_port(); },
        }
    }
}

#[cfg(test)]
mod test {
    use prelude::*;

    use native;
    use os;
    use super::*;

    pub fn stress_factor() -> uint {
        match os::getenv("RUST_TEST_STRESS") {
            Some(val) => from_str::<uint>(val).unwrap(),
            None => 1,
        }
    }

    test!(fn smoke() {
        let (p, c) = Chan::new();
        c.send(1);
        assert_eq!(p.recv(), 1);
    })

    test!(fn drop_full() {
        let (_p, c) = Chan::new();
        c.send(~1);
    })

    test!(fn drop_full_shared() {
        let (_p, c) = Chan::new();
        c.send(~1);
    })

    test!(fn smoke_shared() {
        let (p, c) = Chan::new();
        c.send(1);
        assert_eq!(p.recv(), 1);
        let c = c.clone();
        c.send(1);
        assert_eq!(p.recv(), 1);
    })

    test!(fn smoke_threads() {
        let (p, c) = Chan::new();
        spawn(proc() {
            c.send(1);
        });
        assert_eq!(p.recv(), 1);
    })

    test!(fn smoke_port_gone() {
        let (p, c) = Chan::new();
        drop(p);
        c.send(1);
    } #[should_fail])

    test!(fn smoke_shared_port_gone() {
        let (p, c) = Chan::new();
        drop(p);
        c.send(1);
    } #[should_fail])

    test!(fn smoke_shared_port_gone2() {
        let (p, c) = Chan::new();
        drop(p);
        let c2 = c.clone();
        drop(c);
        c2.send(1);
    } #[should_fail])

    test!(fn port_gone_concurrent() {
        let (p, c) = Chan::new();
        spawn(proc() {
            p.recv();
        });
        loop { c.send(1) }
    } #[should_fail])

    test!(fn port_gone_concurrent_shared() {
        let (p, c) = Chan::new();
        let c1 = c.clone();
        spawn(proc() {
            p.recv();
        });
        loop {
            c.send(1);
            c1.send(1);
        }
    } #[should_fail])

    test!(fn smoke_chan_gone() {
        let (p, c) = Chan::<int>::new();
        drop(c);
        p.recv();
    } #[should_fail])

    test!(fn smoke_chan_gone_shared() {
        let (p, c) = Chan::<()>::new();
        let c2 = c.clone();
        drop(c);
        drop(c2);
        p.recv();
    } #[should_fail])

    test!(fn chan_gone_concurrent() {
        let (p, c) = Chan::new();
        spawn(proc() {
            c.send(1);
            c.send(1);
        });
        loop { p.recv(); }
    } #[should_fail])

    test!(fn stress() {
        let (p, c) = Chan::new();
        spawn(proc() {
            for _ in range(0, 10000) { c.send(1); }
        });
        for _ in range(0, 10000) {
            assert_eq!(p.recv(), 1);
        }
    })

    test!(fn stress_shared() {
        static AMT: uint = 10000;
        static NTHREADS: uint = 8;
        let (p, c) = Chan::<int>::new();
        let (p1, c1) = Chan::new();

        spawn(proc() {
            for _ in range(0, AMT * NTHREADS) {
                assert_eq!(p.recv(), 1);
            }
            match p.try_recv() {
                Data(..) => fail!(),
                _ => {}
            }
            c1.send(());
        });

        for _ in range(0, NTHREADS) {
            let c = c.clone();
            spawn(proc() {
                for _ in range(0, AMT) { c.send(1); }
            });
        }
        p1.recv();
    })

    #[test]
    fn send_from_outside_runtime() {
        let (p, c) = Chan::<int>::new();
        let (p1, c1) = Chan::new();
        let (port, chan) = Chan::new();
        let chan2 = chan.clone();
        spawn(proc() {
            c1.send(());
            for _ in range(0, 40) {
                assert_eq!(p.recv(), 1);
            }
            chan2.send(());
        });
        p1.recv();
        native::task::spawn(proc() {
            for _ in range(0, 40) {
                c.send(1);
            }
            chan.send(());
        });
        port.recv();
        port.recv();
    }

    #[test]
    fn recv_from_outside_runtime() {
        let (p, c) = Chan::<int>::new();
        let (dp, dc) = Chan::new();
        native::task::spawn(proc() {
            for _ in range(0, 40) {
                assert_eq!(p.recv(), 1);
            }
            dc.send(());
        });
        for _ in range(0, 40) {
            c.send(1);
        }
        dp.recv();
    }

    #[test]
    fn no_runtime() {
        let (p1, c1) = Chan::<int>::new();
        let (p2, c2) = Chan::<int>::new();
        let (port, chan) = Chan::new();
        let chan2 = chan.clone();
        native::task::spawn(proc() {
            assert_eq!(p1.recv(), 1);
            c2.send(2);
            chan2.send(());
        });
        native::task::spawn(proc() {
            c1.send(1);
            assert_eq!(p2.recv(), 2);
            chan.send(());
        });
        port.recv();
        port.recv();
    }

    test!(fn oneshot_single_thread_close_port_first() {
        // Simple test of closing without sending
        let (port, _chan) = Chan::<int>::new();
        { let _p = port; }
    })

    test!(fn oneshot_single_thread_close_chan_first() {
        // Simple test of closing without sending
        let (_port, chan) = Chan::<int>::new();
        { let _c = chan; }
    })

    test!(fn oneshot_single_thread_send_port_close() {
        // Testing that the sender cleans up the payload if receiver is closed
        let (port, chan) = Chan::<~int>::new();
        { let _p = port; }
        chan.send(~0);
    } #[should_fail])

    test!(fn oneshot_single_thread_recv_chan_close() {
        // Receiving on a closed chan will fail
        let res = task::try(proc() {
            let (port, chan) = Chan::<~int>::new();
            { let _c = chan; }
            port.recv();
        });
        // What is our res?
        assert!(res.is_err());
    })

    test!(fn oneshot_single_thread_send_then_recv() {
        let (port, chan) = Chan::<~int>::new();
        chan.send(~10);
        assert!(port.recv() == ~10);
    })

    test!(fn oneshot_single_thread_try_send_open() {
        let (port, chan) = Chan::<int>::new();
        assert!(chan.try_send(10));
        assert!(port.recv() == 10);
    })

    test!(fn oneshot_single_thread_try_send_closed() {
        let (port, chan) = Chan::<int>::new();
        { let _p = port; }
        assert!(!chan.try_send(10));
    })

    test!(fn oneshot_single_thread_try_recv_open() {
        let (port, chan) = Chan::<int>::new();
        chan.send(10);
        assert!(port.recv_opt() == Some(10));
    })

    test!(fn oneshot_single_thread_try_recv_closed() {
        let (port, chan) = Chan::<int>::new();
        { let _c = chan; }
        assert!(port.recv_opt() == None);
    })

    test!(fn oneshot_single_thread_peek_data() {
        let (port, chan) = Chan::<int>::new();
        assert_eq!(port.try_recv(), Empty)
        chan.send(10);
        assert_eq!(port.try_recv(), Data(10));
    })

    test!(fn oneshot_single_thread_peek_close() {
        let (port, chan) = Chan::<int>::new();
        { let _c = chan; }
        assert_eq!(port.try_recv(), Disconnected);
        assert_eq!(port.try_recv(), Disconnected);
    })

    test!(fn oneshot_single_thread_peek_open() {
        let (port, _chan) = Chan::<int>::new();
        assert_eq!(port.try_recv(), Empty);
    })

    test!(fn oneshot_multi_task_recv_then_send() {
        let (port, chan) = Chan::<~int>::new();
        spawn(proc() {
            assert!(port.recv() == ~10);
        });

        chan.send(~10);
    })

    test!(fn oneshot_multi_task_recv_then_close() {
        let (port, chan) = Chan::<~int>::new();
        spawn(proc() {
            let _chan = chan;
        });
        let res = task::try(proc() {
            assert!(port.recv() == ~10);
        });
        assert!(res.is_err());
    })

    test!(fn oneshot_multi_thread_close_stress() {
        for _ in range(0, stress_factor()) {
            let (port, chan) = Chan::<int>::new();
            spawn(proc() {
                let _p = port;
            });
            let _chan = chan;
        }
    })

    test!(fn oneshot_multi_thread_send_close_stress() {
        for _ in range(0, stress_factor()) {
            let (port, chan) = Chan::<int>::new();
            spawn(proc() {
                let _p = port;
            });
            let _ = task::try(proc() {
                chan.send(1);
            });
        }
    })

    test!(fn oneshot_multi_thread_recv_close_stress() {
        for _ in range(0, stress_factor()) {
            let (port, chan) = Chan::<int>::new();
            spawn(proc() {
                let port = port;
                let res = task::try(proc() {
                    port.recv();
                });
                assert!(res.is_err());
            });
            spawn(proc() {
                let chan = chan;
                spawn(proc() {
                    let _chan = chan;
                });
            });
        }
    })

    test!(fn oneshot_multi_thread_send_recv_stress() {
        for _ in range(0, stress_factor()) {
            let (port, chan) = Chan::<~int>::new();
            spawn(proc() {
                chan.send(~10);
            });
            spawn(proc() {
                assert!(port.recv() == ~10);
            });
        }
    })

    test!(fn stream_send_recv_stress() {
        for _ in range(0, stress_factor()) {
            let (port, chan) = Chan::<~int>::new();

            send(chan, 0);
            recv(port, 0);

            fn send(chan: Chan<~int>, i: int) {
                if i == 10 { return }

                spawn(proc() {
                    chan.send(~i);
                    send(chan, i + 1);
                });
            }

            fn recv(port: Port<~int>, i: int) {
                if i == 10 { return }

                spawn(proc() {
                    assert!(port.recv() == ~i);
                    recv(port, i + 1);
                });
            }
        }
    })

    test!(fn recv_a_lot() {
        // Regression test that we don't run out of stack in scheduler context
        let (port, chan) = Chan::new();
        for _ in range(0, 10000) { chan.send(()); }
        for _ in range(0, 10000) { port.recv(); }
    })

    test!(fn shared_chan_stress() {
        let (port, chan) = Chan::new();
        let total = stress_factor() + 100;
        for _ in range(0, total) {
            let chan_clone = chan.clone();
            spawn(proc() {
                chan_clone.send(());
            });
        }

        for _ in range(0, total) {
            port.recv();
        }
    })

    test!(fn test_nested_recv_iter() {
        let (port, chan) = Chan::<int>::new();
        let (total_port, total_chan) = Chan::<int>::new();

        spawn(proc() {
            let mut acc = 0;
            for x in port.iter() {
                acc += x;
            }
            total_chan.send(acc);
        });

        chan.send(3);
        chan.send(1);
        chan.send(2);
        drop(chan);
        assert_eq!(total_port.recv(), 6);
    })

    test!(fn test_recv_iter_break() {
        let (port, chan) = Chan::<int>::new();
        let (count_port, count_chan) = Chan::<int>::new();

        spawn(proc() {
            let mut count = 0;
            for x in port.iter() {
                if count >= 3 {
                    break;
                } else {
                    count += x;
                }
            }
            count_chan.send(count);
        });

        chan.send(2);
        chan.send(2);
        chan.send(2);
        chan.try_send(2);
        drop(chan);
        assert_eq!(count_port.recv(), 4);
    })

    test!(fn try_recv_states() {
        let (p, c) = Chan::<int>::new();
        let (p1, c1) = Chan::<()>::new();
        let (p2, c2) = Chan::<()>::new();
        spawn(proc() {
            p1.recv();
            c.send(1);
            c2.send(());
            p1.recv();
            drop(c);
            c2.send(());
        });

        assert_eq!(p.try_recv(), Empty);
        c1.send(());
        p2.recv();
        assert_eq!(p.try_recv(), Data(1));
        assert_eq!(p.try_recv(), Empty);
        c1.send(());
        p2.recv();
        assert_eq!(p.try_recv(), Disconnected);
    })

    // This bug used to end up in a livelock inside of the Port destructor
    // because the internal state of the Shared port was corrupted
    test!(fn destroy_upgraded_shared_port_when_sender_still_active() {
        let (p, c) = Chan::new();
        let (p1, c2) = Chan::new();
        spawn(proc() {
            p.recv(); // wait on a oneshot port
            drop(p);  // destroy a shared port
            c2.send(());
        });
        // make sure the other task has gone to sleep
        for _ in range(0, 5000) { task::deschedule(); }

        // upgrade to a shared chan and send a message
        let t = c.clone();
        drop(c);
        t.send(());

        // wait for the child task to exit before we exit
        p1.recv();
    })
}
