// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Synchronous channels/ports
///
/// This channel implementation differs significantly from the asynchronous
/// implementations found next to it (oneshot/stream/share). This is an
/// implementation of a synchronous, bounded buffer channel.
///
/// Each channel is created with some amount of backing buffer, and sends will
/// *block* until buffer space becomes available. A buffer size of 0 is valid,
/// which means that every successful send is paired with a successful recv.
///
/// This flavor of channels defines a new `send_opt` method for channels which
/// is the method by which a message is sent but the task does not fail if it
/// cannot be delivered.
///
/// Another major difference is that send() will *always* return back the data
/// if it couldn't be sent. This is because it is deterministically known when
/// the data is received and when it is not received.
///
/// Implementation-wise, it can all be summed up with "use a mutex plus some
/// logic". The mutex used here is an OS native mutex, meaning that no user code
/// is run inside of the mutex (to prevent context switching). This
/// implementation shares almost all code for the buffered and unbuffered cases
/// of a synchronous channel. There are a few branches for the unbuffered case,
/// but they're mostly just relevant to blocking senders.

use core::prelude::*;

use alloc::boxed::Box;
use collections::Vec;
use collections::Collection;
use core::mem;
use core::ty::Unsafe;
use rustrt::local::Local;
use rustrt::mutex::{NativeMutex, LockGuard};
use rustrt::task::{Task, BlockedTask};

use atomics;

pub struct Packet<T> {
    /// Only field outside of the mutex. Just done for kicks, but mainly because
    /// the other shared channel already had the code implemented
    channels: atomics::AtomicUint,

    /// The state field is protected by this mutex
    lock: NativeMutex,
    state: Unsafe<State<T>>,
}

struct State<T> {
    disconnected: bool, // Is the channel disconnected yet?
    queue: Queue,       // queue of senders waiting to send data
    blocker: Blocker,   // currently blocked task on this channel
    buf: Buffer<T>,     // storage for buffered messages
    cap: uint,          // capacity of this channel

    /// A curious flag used to indicate whether a sender failed or succeeded in
    /// blocking. This is used to transmit information back to the task that it
    /// must dequeue its message from the buffer because it was not received.
    /// This is only relevant in the 0-buffer case. This obviously cannot be
    /// safely constructed, but it's guaranteed to always have a valid pointer
    /// value.
    canceled: Option<&'static mut bool>,
}

/// Possible flavors of tasks who can be blocked on this channel.
enum Blocker {
    BlockedSender(BlockedTask),
    BlockedReceiver(BlockedTask),
    NoneBlocked
}

/// Simple queue for threading tasks together. Nodes are stack-allocated, so
/// this structure is not safe at all
struct Queue {
    head: *mut Node,
    tail: *mut Node,
}

struct Node {
    task: Option<BlockedTask>,
    next: *mut Node,
}

/// A simple ring-buffer
struct Buffer<T> {
    buf: Vec<Option<T>>,
    start: uint,
    size: uint,
}

#[deriving(Show)]
pub enum Failure {
    Empty,
    Disconnected,
}

/// Atomically blocks the current task, placing it into `slot`, unlocking `lock`
/// in the meantime. This re-locks the mutex upon returning.
fn wait(slot: &mut Blocker, f: fn(BlockedTask) -> Blocker,
        lock: &NativeMutex) {
    let me: Box<Task> = Local::take();
    me.deschedule(1, |task| {
        match mem::replace(slot, f(task)) {
            NoneBlocked => {}
            _ => unreachable!(),
        }
        unsafe { lock.unlock_noguard(); }
        Ok(())
    });
    unsafe { lock.lock_noguard(); }
}

/// Wakes up a task, dropping the lock at the correct time
fn wakeup(task: BlockedTask, guard: LockGuard) {
    // We need to be careful to wake up the waiting task *outside* of the mutex
    // in case it incurs a context switch.
    mem::drop(guard);
    task.wake().map(|t| t.reawaken());
}

impl<T: Send> Packet<T> {
    pub fn new(cap: uint) -> Packet<T> {
        Packet {
            channels: atomics::AtomicUint::new(1),
            lock: unsafe { NativeMutex::new() },
            state: Unsafe::new(State {
                disconnected: false,
                blocker: NoneBlocked,
                cap: cap,
                canceled: None,
                queue: Queue {
                    head: 0 as *mut Node,
                    tail: 0 as *mut Node,
                },
                buf: Buffer {
                    buf: Vec::from_fn(cap + if cap == 0 {1} else {0}, |_| None),
                    start: 0,
                    size: 0,
                },
            }),
        }
    }

    // Locks this channel, returning a guard for the state and the mutable state
    // itself. Care should be taken to ensure that the state does not escape the
    // guard!
    //
    // Note that we're ok promoting an & reference to an &mut reference because
    // the lock ensures that we're the only ones in the world with a pointer to
    // the state.
    fn lock<'a>(&'a self) -> (LockGuard<'a>, &'a mut State<T>) {
        unsafe {
            let guard = self.lock.lock();
            (guard, &mut *self.state.get())
        }
    }

    pub fn send(&self, t: T) -> Result<(), T> {
        let (guard, state) = self.lock();

        // wait for a slot to become available, and enqueue the data
        while !state.disconnected && state.buf.size() == state.buf.cap() {
            state.queue.enqueue(&self.lock);
        }
        if state.disconnected { return Err(t) }
        state.buf.enqueue(t);

        match mem::replace(&mut state.blocker, NoneBlocked) {
            // if our capacity is 0, then we need to wait for a receiver to be
            // available to take our data. After waiting, we check again to make
            // sure the port didn't go away in the meantime. If it did, we need
            // to hand back our data.
            NoneBlocked if state.cap == 0 => {
                let mut canceled = false;
                assert!(state.canceled.is_none());
                state.canceled = Some(unsafe { mem::transmute(&mut canceled) });
                wait(&mut state.blocker, BlockedSender, &self.lock);
                if canceled {Err(state.buf.dequeue())} else {Ok(())}
            }

            // success, we buffered some data
            NoneBlocked => Ok(()),

            // success, someone's about to receive our buffered data.
            BlockedReceiver(task) => { wakeup(task, guard); Ok(()) }

            BlockedSender(..) => fail!("lolwut"),
        }
    }

    pub fn try_send(&self, t: T) -> Result<(), super::TrySendError<T>> {
        let (guard, state) = self.lock();
        if state.disconnected {
            Err(super::RecvDisconnected(t))
        } else if state.buf.size() == state.buf.cap() {
            Err(super::Full(t))
        } else if state.cap == 0 {
            // With capacity 0, even though we have buffer space we can't
            // transfer the data unless there's a receiver waiting.
            match mem::replace(&mut state.blocker, NoneBlocked) {
                NoneBlocked => Err(super::Full(t)),
                BlockedSender(..) => unreachable!(),
                BlockedReceiver(task) => {
                    state.buf.enqueue(t);
                    wakeup(task, guard);
                    Ok(())
                }
            }
        } else {
            // If the buffer has some space and the capacity isn't 0, then we
            // just enqueue the data for later retrieval, ensuring to wake up
            // any blocked receiver if there is one.
            assert!(state.buf.size() < state.buf.cap());
            state.buf.enqueue(t);
            match mem::replace(&mut state.blocker, NoneBlocked) {
                BlockedReceiver(task) => wakeup(task, guard),
                NoneBlocked => {}
                BlockedSender(..) => unreachable!(),
            }
            Ok(())
        }
    }

    // Receives a message from this channel
    //
    // When reading this, remember that there can only ever be one receiver at
    // time.
    pub fn recv(&self) -> Result<T, ()> {
        let (guard, state) = self.lock();

        // Wait for the buffer to have something in it. No need for a while loop
        // because we're the only receiver.
        let mut waited = false;
        if !state.disconnected && state.buf.size() == 0 {
            wait(&mut state.blocker, BlockedReceiver, &self.lock);
            waited = true;
        }
        if state.disconnected && state.buf.size() == 0 { return Err(()) }

        // Pick up the data, wake up our neighbors, and carry on
        assert!(state.buf.size() > 0);
        let ret = state.buf.dequeue();
        self.wakeup_senders(waited, guard, state);
        return Ok(ret);
    }

    pub fn try_recv(&self) -> Result<T, Failure> {
        let (guard, state) = self.lock();

        // Easy cases first
        if state.disconnected { return Err(Disconnected) }
        if state.buf.size() == 0 { return Err(Empty) }

        // Be sure to wake up neighbors
        let ret = Ok(state.buf.dequeue());
        self.wakeup_senders(false, guard, state);

        return ret;
    }

    // Wake up pending senders after some data has been received
    //
    // * `waited` - flag if the receiver blocked to receive some data, or if it
    //              just picked up some data on the way out
    // * `guard` - the lock guard that is held over this channel's lock
    fn wakeup_senders(&self, waited: bool,
                      guard: LockGuard,
                      state: &mut State<T>) {
        let pending_sender1: Option<BlockedTask> = state.queue.dequeue();

        // If this is a no-buffer channel (cap == 0), then if we didn't wait we
        // need to ACK the sender. If we waited, then the sender waking us up
        // was already the ACK.
        let pending_sender2 = if state.cap == 0 && !waited {
            match mem::replace(&mut state.blocker, NoneBlocked) {
                NoneBlocked => None,
                BlockedReceiver(..) => unreachable!(),
                BlockedSender(task) => {
                    state.canceled.take();
                    Some(task)
                }
            }
        } else {
            None
        };
        mem::drop((state, guard));

        // only outside of the lock do we wake up the pending tasks
        pending_sender1.map(|t| t.wake().map(|t| t.reawaken()));
        pending_sender2.map(|t| t.wake().map(|t| t.reawaken()));
    }

    // Prepares this shared packet for a channel clone, essentially just bumping
    // a refcount.
    pub fn clone_chan(&self) {
        self.channels.fetch_add(1, atomics::SeqCst);
    }

    pub fn drop_chan(&self) {
        // Only flag the channel as disconnected if we're the last channel
        match self.channels.fetch_sub(1, atomics::SeqCst) {
            1 => {}
            _ => return
        }

        // Not much to do other than wake up a receiver if one's there
        let (guard, state) = self.lock();
        if state.disconnected { return }
        state.disconnected = true;
        match mem::replace(&mut state.blocker, NoneBlocked) {
            NoneBlocked => {}
            BlockedSender(..) => unreachable!(),
            BlockedReceiver(task) => wakeup(task, guard),
        }
    }

    pub fn drop_port(&self) {
        let (guard, state) = self.lock();

        if state.disconnected { return }
        state.disconnected = true;

        // If the capacity is 0, then the sender may want its data back after
        // we're disconnected. Otherwise it's now our responsibility to destroy
        // the buffered data. As with many other portions of this code, this
        // needs to be careful to destroy the data *outside* of the lock to
        // prevent deadlock.
        let _data = if state.cap != 0 {
            mem::replace(&mut state.buf.buf, Vec::new())
        } else {
            Vec::new()
        };
        let mut queue = mem::replace(&mut state.queue, Queue {
            head: 0 as *mut Node,
            tail: 0 as *mut Node,
        });

        let waiter = match mem::replace(&mut state.blocker, NoneBlocked) {
            NoneBlocked => None,
            BlockedSender(task) => {
                *state.canceled.take_unwrap() = true;
                Some(task)
            }
            BlockedReceiver(..) => unreachable!(),
        };
        mem::drop((state, guard));

        loop {
            match queue.dequeue() {
                Some(task) => { task.wake().map(|t| t.reawaken()); }
                None => break,
            }
        }
        waiter.map(|t| t.wake().map(|t| t.reawaken()));
    }

    ////////////////////////////////////////////////////////////////////////////
    // select implementation
    ////////////////////////////////////////////////////////////////////////////

    // If Ok, the value is whether this port has data, if Err, then the upgraded
    // port needs to be checked instead of this one.
    pub fn can_recv(&self) -> bool {
        let (_g, state) = self.lock();
        state.disconnected || state.buf.size() > 0
    }

    // Attempts to start selection on this port. This can either succeed or fail
    // because there is data waiting.
    pub fn start_selection(&self, task: BlockedTask) -> Result<(), BlockedTask>{
        let (_g, state) = self.lock();
        if state.disconnected || state.buf.size() > 0 {
            Err(task)
        } else {
            match mem::replace(&mut state.blocker, BlockedReceiver(task)) {
                NoneBlocked => {}
                BlockedSender(..) => unreachable!(),
                BlockedReceiver(..) => unreachable!(),
            }
            Ok(())
        }
    }

    // Remove a previous selecting task from this port. This ensures that the
    // blocked task will no longer be visible to any other threads.
    //
    // The return value indicates whether there's data on this port.
    pub fn abort_selection(&self) -> bool {
        let (_g, state) = self.lock();
        match mem::replace(&mut state.blocker, NoneBlocked) {
            NoneBlocked => true,
            BlockedSender(task) => {
                state.blocker = BlockedSender(task);
                true
            }
            BlockedReceiver(task) => { task.trash(); false }
        }
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for Packet<T> {
    fn drop(&mut self) {
        assert_eq!(self.channels.load(atomics::SeqCst), 0);
        let (_g, state) = self.lock();
        assert!(state.queue.dequeue().is_none());
        assert!(state.canceled.is_none());
    }
}


////////////////////////////////////////////////////////////////////////////////
// Buffer, a simple ring buffer backed by Vec<T>
////////////////////////////////////////////////////////////////////////////////

impl<T> Buffer<T> {
    fn enqueue(&mut self, t: T) {
        let pos = (self.start + self.size) % self.buf.len();
        self.size += 1;
        let prev = mem::replace(self.buf.get_mut(pos), Some(t));
        assert!(prev.is_none());
    }

    fn dequeue(&mut self) -> T {
        let start = self.start;
        self.size -= 1;
        self.start = (self.start + 1) % self.buf.len();
        self.buf.get_mut(start).take_unwrap()
    }

    fn size(&self) -> uint { self.size }
    fn cap(&self) -> uint { self.buf.len() }
}

////////////////////////////////////////////////////////////////////////////////
// Queue, a simple queue to enqueue tasks with (stack-allocated nodes)
////////////////////////////////////////////////////////////////////////////////

impl Queue {
    fn enqueue(&mut self, lock: &NativeMutex) {
        let task: Box<Task> = Local::take();
        let mut node = Node {
            task: None,
            next: 0 as *mut Node,
        };
        task.deschedule(1, |task| {
            node.task = Some(task);
            if self.tail.is_null() {
                self.head = &mut node as *mut Node;
                self.tail = &mut node as *mut Node;
            } else {
                unsafe {
                    (*self.tail).next = &mut node as *mut Node;
                    self.tail = &mut node as *mut Node;
                }
            }
            unsafe { lock.unlock_noguard(); }
            Ok(())
        });
        unsafe { lock.lock_noguard(); }
        assert!(node.next.is_null());
    }

    fn dequeue(&mut self) -> Option<BlockedTask> {
        if self.head.is_null() {
            return None
        }
        let node = self.head;
        self.head = unsafe { (*node).next };
        if self.head.is_null() {
            self.tail = 0 as *mut Node;
        }
        unsafe {
            (*node).next = 0 as *mut Node;
            Some((*node).task.take_unwrap())
        }
    }
}
