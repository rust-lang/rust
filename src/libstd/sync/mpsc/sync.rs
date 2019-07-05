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
/// is the method by which a message is sent but the thread does not panic if it
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

pub use self::Failure::*;
use self::Blocker::*;

use core::intrinsics::abort;
use core::isize;
use core::mem;
use core::ptr;

use crate::sync::atomic::{Ordering, AtomicUsize};
use crate::sync::mpsc::blocking::{self, WaitToken, SignalToken};
use crate::sync::{Mutex, MutexGuard};
use crate::time::Instant;

const MAX_REFCOUNT: usize = (isize::MAX) as usize;

pub struct Packet<T> {
    /// Only field outside of the mutex. Just done for kicks, but mainly because
    /// the other shared channel already had the code implemented
    channels: AtomicUsize,

    lock: Mutex<State<T>>,
}

unsafe impl<T: Send> Send for Packet<T> { }

unsafe impl<T: Send> Sync for Packet<T> { }

struct State<T> {
    disconnected: bool, // Is the channel disconnected yet?
    queue: Queue,       // queue of senders waiting to send data
    blocker: Blocker,   // currently blocked thread on this channel
    buf: Buffer<T>,     // storage for buffered messages
    cap: usize,         // capacity of this channel

    /// A curious flag used to indicate whether a sender failed or succeeded in
    /// blocking. This is used to transmit information back to the thread that it
    /// must dequeue its message from the buffer because it was not received.
    /// This is only relevant in the 0-buffer case. This obviously cannot be
    /// safely constructed, but it's guaranteed to always have a valid pointer
    /// value.
    canceled: Option<&'static mut bool>,
}

unsafe impl<T: Send> Send for State<T> {}

/// Possible flavors of threads who can be blocked on this channel.
enum Blocker {
    BlockedSender(SignalToken),
    BlockedReceiver(SignalToken),
    NoneBlocked
}

/// Simple queue for threading threads together. Nodes are stack-allocated, so
/// this structure is not safe at all
struct Queue {
    head: *mut Node,
    tail: *mut Node,
}

struct Node {
    token: Option<SignalToken>,
    next: *mut Node,
}

unsafe impl Send for Node {}

/// A simple ring-buffer
struct Buffer<T> {
    buf: Vec<Option<T>>,
    start: usize,
    size: usize,
}

#[derive(Debug)]
pub enum Failure {
    Empty,
    Disconnected,
}

/// Atomically blocks the current thread, placing it into `slot`, unlocking `lock`
/// in the meantime. This re-locks the mutex upon returning.
fn wait<'a, 'b, T>(lock: &'a Mutex<State<T>>,
                   mut guard: MutexGuard<'b, State<T>>,
                   f: fn(SignalToken) -> Blocker)
                   -> MutexGuard<'a, State<T>>
{
    let (wait_token, signal_token) = blocking::tokens();
    match mem::replace(&mut guard.blocker, f(signal_token)) {
        NoneBlocked => {}
        _ => unreachable!(),
    }
    drop(guard);         // unlock
    wait_token.wait();   // block
    lock.lock().unwrap() // relock
}

/// Same as wait, but waiting at most until `deadline`.
fn wait_timeout_receiver<'a, 'b, T>(lock: &'a Mutex<State<T>>,
                                    deadline: Instant,
                                    mut guard: MutexGuard<'b, State<T>>,
                                    success: &mut bool)
                                    -> MutexGuard<'a, State<T>>
{
    let (wait_token, signal_token) = blocking::tokens();
    match mem::replace(&mut guard.blocker, BlockedReceiver(signal_token)) {
        NoneBlocked => {}
        _ => unreachable!(),
    }
    drop(guard);         // unlock
    *success = wait_token.wait_max_until(deadline);   // block
    let mut new_guard = lock.lock().unwrap(); // relock
    if !*success {
        abort_selection(&mut new_guard);
    }
    new_guard
}

fn abort_selection<T>(guard: &mut MutexGuard<'_, State<T>>) -> bool {
    match mem::replace(&mut guard.blocker, NoneBlocked) {
        NoneBlocked => true,
        BlockedSender(token) => {
            guard.blocker = BlockedSender(token);
            true
        }
        BlockedReceiver(token) => { drop(token); false }
    }
}

/// Wakes up a thread, dropping the lock at the correct time
fn wakeup<T>(token: SignalToken, guard: MutexGuard<'_, State<T>>) {
    // We need to be careful to wake up the waiting thread *outside* of the mutex
    // in case it incurs a context switch.
    drop(guard);
    token.signal();
}

impl<T> Packet<T> {
    pub fn new(cap: usize) -> Packet<T> {
        Packet {
            channels: AtomicUsize::new(1),
            lock: Mutex::new(State {
                disconnected: false,
                blocker: NoneBlocked,
                cap,
                canceled: None,
                queue: Queue {
                    head: ptr::null_mut(),
                    tail: ptr::null_mut(),
                },
                buf: Buffer {
                    buf: (0..cap + if cap == 0 {1} else {0}).map(|_| None).collect(),
                    start: 0,
                    size: 0,
                },
            }),
        }
    }

    // wait until a send slot is available, returning locked access to
    // the channel state.
    fn acquire_send_slot(&self) -> MutexGuard<'_, State<T>> {
        let mut node = Node { token: None, next: ptr::null_mut() };
        loop {
            let mut guard = self.lock.lock().unwrap();
            // are we ready to go?
            if guard.disconnected || guard.buf.size() < guard.buf.cap() {
                return guard;
            }
            // no room; actually block
            let wait_token = guard.queue.enqueue(&mut node);
            drop(guard);
            wait_token.wait();
        }
    }

    pub fn send(&self, t: T) -> Result<(), T> {
        let mut guard = self.acquire_send_slot();
        if guard.disconnected { return Err(t) }
        guard.buf.enqueue(t);

        match mem::replace(&mut guard.blocker, NoneBlocked) {
            // if our capacity is 0, then we need to wait for a receiver to be
            // available to take our data. After waiting, we check again to make
            // sure the port didn't go away in the meantime. If it did, we need
            // to hand back our data.
            NoneBlocked if guard.cap == 0 => {
                let mut canceled = false;
                assert!(guard.canceled.is_none());
                guard.canceled = Some(unsafe { mem::transmute(&mut canceled) });
                let mut guard = wait(&self.lock, guard, BlockedSender);
                if canceled {Err(guard.buf.dequeue())} else {Ok(())}
            }

            // success, we buffered some data
            NoneBlocked => Ok(()),

            // success, someone's about to receive our buffered data.
            BlockedReceiver(token) => { wakeup(token, guard); Ok(()) }

            BlockedSender(..) => panic!("lolwut"),
        }
    }

    pub fn try_send(&self, t: T) -> Result<(), super::TrySendError<T>> {
        let mut guard = self.lock.lock().unwrap();
        if guard.disconnected {
            Err(super::TrySendError::Disconnected(t))
        } else if guard.buf.size() == guard.buf.cap() {
            Err(super::TrySendError::Full(t))
        } else if guard.cap == 0 {
            // With capacity 0, even though we have buffer space we can't
            // transfer the data unless there's a receiver waiting.
            match mem::replace(&mut guard.blocker, NoneBlocked) {
                NoneBlocked => Err(super::TrySendError::Full(t)),
                BlockedSender(..) => unreachable!(),
                BlockedReceiver(token) => {
                    guard.buf.enqueue(t);
                    wakeup(token, guard);
                    Ok(())
                }
            }
        } else {
            // If the buffer has some space and the capacity isn't 0, then we
            // just enqueue the data for later retrieval, ensuring to wake up
            // any blocked receiver if there is one.
            assert!(guard.buf.size() < guard.buf.cap());
            guard.buf.enqueue(t);
            match mem::replace(&mut guard.blocker, NoneBlocked) {
                BlockedReceiver(token) => wakeup(token, guard),
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
    pub fn recv(&self, deadline: Option<Instant>) -> Result<T, Failure> {
        let mut guard = self.lock.lock().unwrap();

        let mut woke_up_after_waiting = false;
        // Wait for the buffer to have something in it. No need for a
        // while loop because we're the only receiver.
        if !guard.disconnected && guard.buf.size() == 0 {
            if let Some(deadline) = deadline {
                guard = wait_timeout_receiver(&self.lock,
                                              deadline,
                                              guard,
                                              &mut woke_up_after_waiting);
            } else {
                guard = wait(&self.lock, guard, BlockedReceiver);
                woke_up_after_waiting = true;
            }
        }

        // N.B., channel could be disconnected while waiting, so the order of
        // these conditionals is important.
        if guard.disconnected && guard.buf.size() == 0 {
            return Err(Disconnected);
        }

        // Pick up the data, wake up our neighbors, and carry on
        assert!(guard.buf.size() > 0 || (deadline.is_some() && !woke_up_after_waiting));

        if guard.buf.size() == 0 { return Err(Empty); }

        let ret = guard.buf.dequeue();
        self.wakeup_senders(woke_up_after_waiting, guard);
        Ok(ret)
    }

    pub fn try_recv(&self) -> Result<T, Failure> {
        let mut guard = self.lock.lock().unwrap();

        // Easy cases first
        if guard.disconnected && guard.buf.size() == 0 { return Err(Disconnected) }
        if guard.buf.size() == 0 { return Err(Empty) }

        // Be sure to wake up neighbors
        let ret = Ok(guard.buf.dequeue());
        self.wakeup_senders(false, guard);
        ret
    }

    // Wake up pending senders after some data has been received
    //
    // * `waited` - flag if the receiver blocked to receive some data, or if it
    //              just picked up some data on the way out
    // * `guard` - the lock guard that is held over this channel's lock
    fn wakeup_senders(&self, waited: bool, mut guard: MutexGuard<'_, State<T>>) {
        let pending_sender1: Option<SignalToken> = guard.queue.dequeue();

        // If this is a no-buffer channel (cap == 0), then if we didn't wait we
        // need to ACK the sender. If we waited, then the sender waking us up
        // was already the ACK.
        let pending_sender2 = if guard.cap == 0 && !waited {
            match mem::replace(&mut guard.blocker, NoneBlocked) {
                NoneBlocked => None,
                BlockedReceiver(..) => unreachable!(),
                BlockedSender(token) => {
                    guard.canceled.take();
                    Some(token)
                }
            }
        } else {
            None
        };
        mem::drop(guard);

        // only outside of the lock do we wake up the pending threads
        pending_sender1.map(|t| t.signal());
        pending_sender2.map(|t| t.signal());
    }

    // Prepares this shared packet for a channel clone, essentially just bumping
    // a refcount.
    pub fn clone_chan(&self) {
        let old_count = self.channels.fetch_add(1, Ordering::SeqCst);

        // See comments on Arc::clone() on why we do this (for `mem::forget`).
        if old_count > MAX_REFCOUNT {
            unsafe {
                abort();
            }
        }
    }

    pub fn drop_chan(&self) {
        // Only flag the channel as disconnected if we're the last channel
        match self.channels.fetch_sub(1, Ordering::SeqCst) {
            1 => {}
            _ => return
        }

        // Not much to do other than wake up a receiver if one's there
        let mut guard = self.lock.lock().unwrap();
        if guard.disconnected { return }
        guard.disconnected = true;
        match mem::replace(&mut guard.blocker, NoneBlocked) {
            NoneBlocked => {}
            BlockedSender(..) => unreachable!(),
            BlockedReceiver(token) => wakeup(token, guard),
        }
    }

    pub fn drop_port(&self) {
        let mut guard = self.lock.lock().unwrap();

        if guard.disconnected { return }
        guard.disconnected = true;

        // If the capacity is 0, then the sender may want its data back after
        // we're disconnected. Otherwise it's now our responsibility to destroy
        // the buffered data. As with many other portions of this code, this
        // needs to be careful to destroy the data *outside* of the lock to
        // prevent deadlock.
        let _data = if guard.cap != 0 {
            mem::take(&mut guard.buf.buf)
        } else {
            Vec::new()
        };
        let mut queue = mem::replace(&mut guard.queue, Queue {
            head: ptr::null_mut(),
            tail: ptr::null_mut(),
        });

        let waiter = match mem::replace(&mut guard.blocker, NoneBlocked) {
            NoneBlocked => None,
            BlockedSender(token) => {
                *guard.canceled.take().unwrap() = true;
                Some(token)
            }
            BlockedReceiver(..) => unreachable!(),
        };
        mem::drop(guard);

        while let Some(token) = queue.dequeue() { token.signal(); }
        waiter.map(|t| t.signal());
    }
}

impl<T> Drop for Packet<T> {
    fn drop(&mut self) {
        assert_eq!(self.channels.load(Ordering::SeqCst), 0);
        let mut guard = self.lock.lock().unwrap();
        assert!(guard.queue.dequeue().is_none());
        assert!(guard.canceled.is_none());
    }
}


////////////////////////////////////////////////////////////////////////////////
// Buffer, a simple ring buffer backed by Vec<T>
////////////////////////////////////////////////////////////////////////////////

impl<T> Buffer<T> {
    fn enqueue(&mut self, t: T) {
        let pos = (self.start + self.size) % self.buf.len();
        self.size += 1;
        let prev = mem::replace(&mut self.buf[pos], Some(t));
        assert!(prev.is_none());
    }

    fn dequeue(&mut self) -> T {
        let start = self.start;
        self.size -= 1;
        self.start = (self.start + 1) % self.buf.len();
        let result = &mut self.buf[start];
        result.take().unwrap()
    }

    fn size(&self) -> usize { self.size }
    fn cap(&self) -> usize { self.buf.len() }
}

////////////////////////////////////////////////////////////////////////////////
// Queue, a simple queue to enqueue threads with (stack-allocated nodes)
////////////////////////////////////////////////////////////////////////////////

impl Queue {
    fn enqueue(&mut self, node: &mut Node) -> WaitToken {
        let (wait_token, signal_token) = blocking::tokens();
        node.token = Some(signal_token);
        node.next = ptr::null_mut();

        if self.tail.is_null() {
            self.head = node as *mut Node;
            self.tail = node as *mut Node;
        } else {
            unsafe {
                (*self.tail).next = node as *mut Node;
                self.tail = node as *mut Node;
            }
        }

        wait_token
    }

    fn dequeue(&mut self) -> Option<SignalToken> {
        if self.head.is_null() {
            return None
        }
        let node = self.head;
        self.head = unsafe { (*node).next };
        if self.head.is_null() {
            self.tail = ptr::null_mut();
        }
        unsafe {
            (*node).next = ptr::null_mut();
            Some((*node).token.take().unwrap())
        }
    }
}
