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
//! This module currently provides three main types:
//!
//! * `Chan`
//! * `Port`
//! * `SharedChan`
//!
//! The `Chan` and `SharedChan` types are used to send data to a `Port`. A
//! `SharedChan` is clone-able such that many tasks can send simultaneously to
//! one receiving port. These communication primitives are *task blocking*, not
//! *thread blocking*. This means that if one task is blocked on a channel,
//! other tasks can continue to make progress.
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
//! There are methods on all of `Chan`, `SharedChan`, and `Port` to perform
//! their respective operations without failing, however.
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
//! let (port, chan) = SharedChan::new();
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
// Rust channels come in two flavors: streams and shared channels. A stream has
// one sender and one receiver while a shared channel could have multiple
// senders. This choice heavily influences the design of the protocol set
// forth for both senders/receivers.
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
use clone::Clone;
use container::Container;
use int;
use iter::Iterator;
use kinds::marker;
use kinds::Send;
use ops::Drop;
use option::{Option, Some, None};
use result::{Ok, Err};
use rt::local::Local;
use rt::task::{Task, BlockedTask};
use rt::thread::Thread;
use sync::atomics::{AtomicInt, AtomicBool, SeqCst, Relaxed};
use vec::OwnedVector;

use spsc = sync::spsc_queue;
use mpsc = sync::mpsc_queue;

pub use self::select::{Select, Handle};

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

///////////////////////////////////////////////////////////////////////////////
// Helper type to abstract ports for channels and shared channels
///////////////////////////////////////////////////////////////////////////////

enum Consumer<T> {
    SPSC(spsc::Consumer<T, Packet>),
    MPSC(mpsc::Consumer<T, Packet>),
}

impl<T: Send> Consumer<T>{
    unsafe fn packet(&self) -> *mut Packet {
        match *self {
            SPSC(ref c) => c.packet(),
            MPSC(ref c) => c.packet(),
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Public structs
///////////////////////////////////////////////////////////////////////////////

/// The receiving-half of Rust's channel type. This half can only be owned by
/// one task
pub struct Port<T> {
    priv queue: Consumer<T>,

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
    priv queue: spsc::Producer<T, Packet>,

    // can't share in an arc
    priv marker: marker::NoFreeze,
}

/// The sending-half of Rust's channel type. This half can be shared among many
/// tasks by creating copies of itself through the `clone` method.
pub struct SharedChan<T> {
    priv queue: mpsc::Producer<T, Packet>,

    // can't share in an arc -- technically this implementation is
    // shareable, but it shouldn't be required to be shareable in an
    // arc
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

///////////////////////////////////////////////////////////////////////////////
// Internal struct definitions
///////////////////////////////////////////////////////////////////////////////

struct Packet {
    cnt: AtomicInt, // How many items are on this channel
    steals: int,    // How many times has a port received without blocking?
    to_wake: Option<BlockedTask>, // Task to wake up

    // This lock is used to wake up native threads blocked in select. The
    // `lock` field is not used because the thread blocking in select must
    // block on only one mutex.
    //selection_lock: Option<UnsafeArc<Mutex>>,

    // The number of channels which are currently using this packet. This is
    // used to reference count shared channels.
    channels: AtomicInt,

    selecting: AtomicBool,
    selection_id: uint,
    select_next: *mut Packet,
    select_prev: *mut Packet,
    recv_cnt: int,
}

///////////////////////////////////////////////////////////////////////////////
// All implementations -- the fun part
///////////////////////////////////////////////////////////////////////////////

static DISCONNECTED: int = int::MIN;
static RESCHED_FREQ: int = 200;

impl Packet {
    fn new() -> Packet {
        Packet {
            cnt: AtomicInt::new(0),
            steals: 0,
            to_wake: None,
            channels: AtomicInt::new(1),

            selecting: AtomicBool::new(false),
            selection_id: 0,
            select_next: 0 as *mut Packet,
            select_prev: 0 as *mut Packet,
            recv_cnt: 0,
        }
    }

    // Increments the channel size count, preserving the disconnected state if
    // the other end has disconnected.
    fn increment(&mut self) -> int {
        match self.cnt.fetch_add(1, SeqCst) {
            DISCONNECTED => {
                // see the comment in 'try' for a shared channel for why this
                // window of "not disconnected" is "ok".
                self.cnt.store(DISCONNECTED, SeqCst);
                DISCONNECTED
            }
            n => n
        }
    }

    // Decrements the reference count of the channel, returning whether the task
    // should block or not. This assumes that the task is ready to sleep in that
    // the `to_wake` field has already been filled in. Once this decrement
    // happens, the task could wake up on the other end.
    //
    // From an implementation perspective, this is also when our "steal count"
    // gets merged into the "channel count". Our steal count is reset to 0 after
    // this function completes.
    //
    // As with increment(), this preserves the disconnected state if the
    // channel is disconnected.
    fn decrement(&mut self) -> bool {
        let steals = self.steals;
        self.steals = 0;
        match self.cnt.fetch_sub(1 + steals, SeqCst) {
            DISCONNECTED => {
                self.cnt.store(DISCONNECTED, SeqCst);
                false
            }
            n => {
                assert!(n >= 0);
                n - steals <= 0
            }
        }
    }

    // Helper function for select, tests whether this port can receive without
    // blocking (obviously not an atomic decision).
    fn can_recv(&self) -> bool {
        let cnt = self.cnt.load(SeqCst);
        cnt == DISCONNECTED || cnt - self.steals > 0
    }

    // This function must have had at least an acquire fence before it to be
    // properly called.
    fn wakeup(&mut self) {
        match self.to_wake.take_unwrap().wake() {
            Some(task) => task.reawaken(),
            None => {}
        }
        self.selecting.store(false, Relaxed);
    }

    // Aborts the selection process for a port. This happens as part of select()
    // once the task has reawoken. This will place the channel back into a
    // consistent state which is ready to be received from again.
    //
    // The method of doing this is a little subtle. These channels have the
    // invariant that if -1 is seen, then to_wake is always Some(..) and should
    // be woken up. This aborting process at least needs to add 1 to the
    // reference count, but that is not guaranteed to make the count positive
    // (our steal count subtraction could mean that after the addition the
    // channel count is still negative).
    //
    // In order to get around this, we force our channel count to go above 0 by
    // adding a large number >= 1 to it. This way no sender will see -1 unless
    // we are indeed blocking. This "extra lump" we took out of the channel
    // becomes our steal count (which will get re-factored into the count on the
    // next blocking recv)
    //
    // The return value of this method is whether there is data on this channel
    // to receive or not.
    fn abort_selection(&mut self, take_to_wake: bool) -> bool {
        // make sure steals + 1 makes the count go non-negative
        let steals = {
            let cnt = self.cnt.load(SeqCst);
            if cnt < 0 && cnt != DISCONNECTED {-cnt} else {0}
        };
        let prev = self.cnt.fetch_add(steals + 1, SeqCst);

        // If we were previously disconnected, then we know for sure that there
        // is no task in to_wake, so just keep going
        if prev == DISCONNECTED {
            assert!(self.to_wake.is_none());
            self.cnt.store(DISCONNECTED, SeqCst);
            self.selecting.store(false, SeqCst);
            true // there is data, that data is that we're disconnected
        } else {
            let cur = prev + steals + 1;
            assert!(cur >= 0);

            // If the previous count was negative, then we just made things go
            // positive, hence we passed the -1 boundary and we're responsible
            // for removing the to_wake() field and trashing it.
            if prev < 0 {
                if take_to_wake {
                    self.to_wake.take_unwrap().trash();
                } else {
                    assert!(self.to_wake.is_none());
                }

                // We woke ourselves up, we're responsible for cancelling
                assert!(self.selecting.load(Relaxed));
                self.selecting.store(false, Relaxed);
            }
            assert_eq!(self.steals, 0);
            self.steals = steals;

            // if we were previously positive, then there's surely data to
            // receive
            prev >= 0
        }
    }

    // Decrement the reference count on a channel. This is called whenever a
    // Chan is dropped and may end up waking up a receiver. It's the receiver's
    // responsibility on the other end to figure out that we've disconnected.
    unsafe fn drop_chan(&mut self) {
        match self.channels.fetch_sub(1, SeqCst) {
            1 => {
                match self.cnt.swap(DISCONNECTED, SeqCst) {
                    -1 => { self.wakeup(); }
                    DISCONNECTED => {}
                    n => { assert!(n >= 0); }
                }
            }
            n if n > 1 => {},
            n => fail!("bad number of channels left {}", n),
        }
    }
}

impl Drop for Packet {
    fn drop(&mut self) {
        unsafe {
            // Note that this load is not only an assert for correctness about
            // disconnection, but also a proper fence before the read of
            // `to_wake`, so this assert cannot be removed with also removing
            // the `to_wake` assert.
            assert_eq!(self.cnt.load(SeqCst), DISCONNECTED);
            assert!(self.to_wake.is_none());
            assert_eq!(self.channels.load(SeqCst), 0);
        }
    }
}

impl<T: Send> Chan<T> {
    /// Creates a new port/channel pair. All data send on the channel returned
    /// will become available on the port as well. See the documentation of
    /// `Port` and `Chan` to see what's possible with them.
    pub fn new() -> (Port<T>, Chan<T>) {
        // arbitrary 128 size cache -- this is just a max cache size, not a
        // maximum buffer size
        let (c, p) = spsc::queue(128, Packet::new());
        let c = SPSC(c);
        (Port { queue: c, marker: marker::NoFreeze },
         Chan { queue: p, marker: marker::NoFreeze })
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
        unsafe {
            let this = cast::transmute_mut(self);
            this.queue.push(t);
            let packet = this.queue.packet();
            match (*packet).increment() {
                // As described above, -1 == wakeup
                -1 => { (*packet).wakeup(); true }
                // Also as above, SPSC queues must be >= -2
                -2 => true,
                // We succeeded if we sent data
                DISCONNECTED => this.queue.is_empty(),
                // In order to prevent starvation of other tasks in situations
                // where a task sends repeatedly without ever receiving, we
                // occassionally yield instead of doing a send immediately.
                // Only doing this if we're doing a rescheduling send, otherwise
                // the caller is expecting not to context switch.
                //
                // Note that we don't unconditionally attempt to yield because
                // the TLS overhead can be a bit much.
                n => {
                    assert!(n >= 0);
                    if n > 0 && n % RESCHED_FREQ == 0 {
                        let task: ~Task = Local::take();
                        task.maybe_yield();
                    }
                    true
                }
            }
        }
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for Chan<T> {
    fn drop(&mut self) {
        unsafe { (*self.queue.packet()).drop_chan(); }
    }
}

impl<T: Send> SharedChan<T> {
    /// Creates a new shared channel and port pair. The purpose of a shared
    /// channel is to be cloneable such that many tasks can send data at the
    /// same time. All data sent on any channel will become available on the
    /// provided port as well.
    pub fn new() -> (Port<T>, SharedChan<T>) {
        let (c, p) = mpsc::queue(Packet::new());
        let c = MPSC(c);
        (Port { queue: c, marker: marker::NoFreeze },
         SharedChan { queue: p, marker: marker::NoFreeze })
    }

    /// Equivalent method to `send` on the `Chan` type (using the same
    /// semantics)
    pub fn send(&self, t: T) {
        if !self.try_send(t) {
            fail!("sending on a closed channel");
        }
    }

    /// Equivalent method to `try_send` on the `Chan` type (using the same
    /// semantics)
    pub fn try_send(&self, t: T) -> bool {
        unsafe {
            // Note that the multiple sender case is a little tricker
            // semantically than the single sender case. The logic for
            // incrementing is "add and if disconnected store disconnected".
            // This could end up leading some senders to believe that there
            // wasn't a disconnect if in fact there was a disconnect. This means
            // that while one thread is attempting to re-store the disconnected
            // states, other threads could walk through merrily incrementing
            // this very-negative disconnected count. To prevent senders from
            // spuriously attempting to send when the channels is actually
            // disconnected, the count has a ranged check here.
            //
            // This is also done for another reason. Remember that the return
            // value of this function is:
            //
            //  `true` == the data *may* be received, this essentially has no
            //            meaning
            //  `false` == the data will *never* be received, this has a lot of
            //             meaning
            //
            // In the SPSC case, we have a check of 'queue.is_empty()' to see
            // whether the data was actually received, but this same condition
            // means nothing in a multi-producer context. As a result, this
            // preflight check serves as the definitive "this will never be
            // received". Once we get beyond this check, we have permanently
            // entered the realm of "this may be received"
            let packet = self.queue.packet();
            if (*packet).cnt.load(Relaxed) < DISCONNECTED + 1024 {
                return false
            }

            let this = cast::transmute_mut(self);
            this.queue.push(t);

            match (*packet).increment() {
                DISCONNECTED => {} // oh well, we tried
                -1 => { (*packet).wakeup(); }
                n => {
                    if n > 0 && n % RESCHED_FREQ == 0 {
                        let task: ~Task = Local::take();
                        task.maybe_yield();
                    }
                }
            }
            true
        }
    }
}

impl<T: Send> Clone for SharedChan<T> {
    fn clone(&self) -> SharedChan<T> {
        unsafe { (*self.queue.packet()).channels.fetch_add(1, SeqCst); }
        SharedChan { queue: self.queue.clone(), marker: marker::NoFreeze }
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for SharedChan<T> {
    fn drop(&mut self) {
        unsafe { (*self.queue.packet()).drop_chan(); }
    }
}

impl<T: Send> Port<T> {
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
        self.try_recv_inc(true)
    }

    fn try_recv_inc(&self, increment: bool) -> TryRecvResult<T> {
        // This is a "best effort" situation, so if a queue is inconsistent just
        // don't worry about it.
        let this = unsafe { cast::transmute_mut(self) };

        // See the comment about yielding on sends, but the same applies here.
        // If a thread is spinning in try_recv we should try
        unsafe {
            let packet = this.queue.packet();
            (*packet).recv_cnt += 1;
            if (*packet).recv_cnt % RESCHED_FREQ == 0 {
                let task: ~Task = Local::take();
                task.maybe_yield();
            }
        }

        let ret = match this.queue {
            SPSC(ref mut queue) => queue.pop(),
            MPSC(ref mut queue) => match queue.pop() {
                mpsc::Data(t) => Some(t),
                mpsc::Empty => None,

                // This is a bit of an interesting case. The channel is
                // reported as having data available, but our pop() has
                // failed due to the queue being in an inconsistent state.
                // This means that there is some pusher somewhere which has
                // yet to complete, but we are guaranteed that a pop will
                // eventually succeed. In this case, we spin in a yield loop
                // because the remote sender should finish their enqueue
                // operation "very quickly".
                //
                // Note that this yield loop does *not* attempt to do a green
                // yield (regardless of the context), but *always* performs an
                // OS-thread yield. The reasoning for this is that the pusher in
                // question which is causing the inconsistent state is
                // guaranteed to *not* be a blocked task (green tasks can't get
                // pre-empted), so it must be on a different OS thread. Also,
                // `try_recv` is normally a "guaranteed no rescheduling" context
                // in a green-thread situation. By yielding control of the
                // thread, we will hopefully allow time for the remote task on
                // the other OS thread to make progress.
                //
                // Avoiding this yield loop would require a different queue
                // abstraction which provides the guarantee that after M
                // pushes have succeeded, at least M pops will succeed. The
                // current queues guarantee that if there are N active
                // pushes, you can pop N times once all N have finished.
                mpsc::Inconsistent => {
                    let data;
                    loop {
                        Thread::yield_now();
                        match queue.pop() {
                            mpsc::Data(t) => { data = t; break }
                            mpsc::Empty => fail!("inconsistent => empty"),
                            mpsc::Inconsistent => {}
                        }
                    }
                    Some(data)
                }
            }
        };
        if increment && ret.is_some() {
            unsafe { (*this.queue.packet()).steals += 1; }
        }
        match ret {
            Some(t) => Data(t),
            None => {
                // It's possible that between the time that we saw the queue was
                // empty and here the other side disconnected. It's also
                // possible for us to see the disconnection here while there is
                // data in the queue. It's pretty backwards-thinking to return
                // Disconnected when there's actually data on the queue, so if
                // we see a disconnected state be sure to check again to be 100%
                // sure that there's no data in the queue.
                let cnt = unsafe { (*this.queue.packet()).cnt.load(Relaxed) };
                if cnt != DISCONNECTED { return Empty }

                let ret = match this.queue {
                    SPSC(ref mut queue) => queue.pop(),
                    MPSC(ref mut queue) => match queue.pop() {
                        mpsc::Data(t) => Some(t),
                        mpsc::Empty => None,
                        mpsc::Inconsistent => {
                            fail!("inconsistent with no senders?!");
                        }
                    }
                };
                match ret {
                    Some(data) => Data(data),
                    None => Disconnected,
                }
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
        // optimistic preflight check (scheduling is expensive)
        match self.try_recv() {
            Empty => {},
            Disconnected => return None,
            Data(t) => return Some(t),
        }

        let packet;
        let this;
        unsafe {
            this = cast::transmute_mut(self);
            packet = this.queue.packet();
            let task: ~Task = Local::take();
            task.deschedule(1, |task| {
                assert!((*packet).to_wake.is_none());
                (*packet).to_wake = Some(task);
                if (*packet).decrement() {
                    Ok(())
                } else {
                    Err((*packet).to_wake.take_unwrap())
                }
            });
        }

        match self.try_recv_inc(false) {
            Data(t) => Some(t),
            Empty => fail!("bug: woke up too soon"),
            Disconnected => None,
        }
    }

    /// Returns an iterator which will block waiting for messages, but never
    /// `fail!`. It will return `None` when the channel has hung up.
    pub fn iter<'a>(&'a self) -> Messages<'a, T> {
        Messages { port: self }
    }
}

impl<'a, T: Send> Iterator<T> for Messages<'a, T> {
    fn next(&mut self) -> Option<T> { self.port.recv_opt() }
}

#[unsafe_destructor]
impl<T: Send> Drop for Port<T> {
    fn drop(&mut self) {
        // All we need to do is store that we're disconnected. If the channel
        // half has already disconnected, then we'll just deallocate everything
        // when the shared packet is deallocated.
        unsafe {
            (*self.queue.packet()).cnt.store(DISCONNECTED, SeqCst);
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
        let (_p, c) = SharedChan::new();
        c.send(~1);
    })

    test!(fn smoke_shared() {
        let (p, c) = SharedChan::new();
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
        let (p, c) = SharedChan::new();
        drop(p);
        c.send(1);
    } #[should_fail])

    test!(fn smoke_shared_port_gone2() {
        let (p, c) = SharedChan::new();
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
        let (p, c) = SharedChan::new();
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
        let (p, c) = SharedChan::<()>::new();
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
        let (p, c) = SharedChan::<int>::new();
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
        let (port, chan) = SharedChan::new();
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
        let (port, chan) = SharedChan::new();
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
        let (port, chan) = SharedChan::new();
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
}
