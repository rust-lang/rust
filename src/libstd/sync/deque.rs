// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A (mostly) lock-free concurrent work-stealing deque
//!
//! This module contains an implementation of the Chase-Lev work stealing deque
//! described in "Dynamic Circular Work-Stealing Deque". The implementation is
//! heavily based on the pseudocode found in the paper.
//!
//! This implementation does not want to have the restriction of a garbage
//! collector for reclamation of buffers, and instead it uses a shared pool of
//! buffers. This shared pool is required for correctness in this
//! implementation.
//!
//! The only lock-synchronized portions of this deque are the buffer allocation
//! and deallocation portions. Otherwise all operations are lock-free.
//!
//! # Example
//!
//!     use std::rt::deque::BufferPool;
//!
//!     let mut pool = BufferPool::new();
//!     let (mut worker, mut stealer) = pool.deque();
//!
//!     // Only the worker may push/pop
//!     worker.push(1);
//!     worker.pop();
//!
//!     // Stealers take data from the other end of the deque
//!     worker.push(1);
//!     stealer.steal();
//!
//!     // Stealers can be cloned to have many stealers stealing in parallel
//!     worker.push(1);
//!     let mut stealer2 = stealer.clone();
//!     stealer2.steal();

// NB: the "buffer pool" strategy is not done for speed, but rather for
//     correctness. For more info, see the comment on `swap_buffer`

// XXX: all atomic operations in this module use a SeqCst ordering. That is
//      probably overkill

use cast;
use clone::Clone;
use iter::{range, Iterator};
use kinds::Send;
use libc;
use mem;
use ops::Drop;
use option::{Option, Some, None};
use ptr;
use ptr::RawPtr;
use sync::arc::UnsafeArc;
use sync::atomics::{AtomicInt, AtomicPtr, SeqCst};
use unstable::sync::Exclusive;
use vec::{OwnedVector, ImmutableVector};

// Once the queue is less than 1/K full, then it will be downsized. Note that
// the deque requires that this number be less than 2.
static K: int = 4;

// Minimum number of bits that a buffer size should be. No buffer will resize to
// under this value, and all deques will initially contain a buffer of this
// size.
//
// The size in question is 1 << MIN_BITS
static MIN_BITS: int = 7;

struct Deque<T> {
    bottom: AtomicInt,
    top: AtomicInt,
    array: AtomicPtr<Buffer<T>>,
    pool: BufferPool<T>,
}

/// Worker half of the work-stealing deque. This worker has exclusive access to
/// one side of the deque, and uses `push` and `pop` method to manipulate it.
///
/// There may only be one worker per deque.
pub struct Worker<T> {
    priv deque: UnsafeArc<Deque<T>>,
}

/// The stealing half of the work-stealing deque. Stealers have access to the
/// opposite end of the deque from the worker, and they only have access to the
/// `steal` method.
pub struct Stealer<T> {
    priv deque: UnsafeArc<Deque<T>>,
}

/// When stealing some data, this is an enumeration of the possible outcomes.
#[deriving(Eq)]
pub enum Stolen<T> {
    /// The deque was empty at the time of stealing
    Empty,
    /// The stealer lost the race for stealing data, and a retry may return more
    /// data.
    Abort,
    /// The stealer has successfully stolen some data.
    Data(T),
}

/// The allocation pool for buffers used by work-stealing deques. Right now this
/// structure is used for reclamation of memory after it is no longer in use by
/// deques.
///
/// This data structure is protected by a mutex, but it is rarely used. Deques
/// will only use this structure when allocating a new buffer or deallocating a
/// previous one.
pub struct BufferPool<T> {
    priv pool: Exclusive<~[~Buffer<T>]>,
}

/// An internal buffer used by the chase-lev deque. This structure is actually
/// implemented as a circular buffer, and is used as the intermediate storage of
/// the data in the deque.
///
/// This type is implemented with *T instead of ~[T] for two reasons:
///
///   1. There is nothing safe about using this buffer. This easily allows the
///      same value to be read twice in to rust, and there is nothing to
///      prevent this. The usage by the deque must ensure that one of the
///      values is forgotten. Furthermore, we only ever want to manually run
///      destructors for values in this buffer (on drop) because the bounds
///      are defined by the deque it's owned by.
///
///   2. We can certainly avoid bounds checks using *T instead of ~[T], although
///      LLVM is probably pretty good at doing this already.
struct Buffer<T> {
    storage: *T,
    log_size: int,
}

impl<T: Send> BufferPool<T> {
    /// Allocates a new buffer pool which in turn can be used to allocate new
    /// deques.
    pub fn new() -> BufferPool<T> {
        BufferPool { pool: Exclusive::new(~[]) }
    }

    /// Allocates a new work-stealing deque which will send/receiving memory to
    /// and from this buffer pool.
    pub fn deque(&mut self) -> (Worker<T>, Stealer<T>) {
        let (a, b) = UnsafeArc::new2(Deque::new(self.clone()));
        (Worker { deque: a }, Stealer { deque: b })
    }

    fn alloc(&mut self, bits: int) -> ~Buffer<T> {
        unsafe {
            self.pool.with(|pool| {
                match pool.iter().position(|x| x.size() >= (1 << bits)) {
                    Some(i) => pool.remove(i).unwrap(),
                    None => ~Buffer::new(bits)
                }
            })
        }
    }

    fn free(&mut self, buf: ~Buffer<T>) {
        unsafe {
            let mut buf = Some(buf);
            self.pool.with(|pool| {
                let buf = buf.take_unwrap();
                match pool.iter().position(|v| v.size() > buf.size()) {
                    Some(i) => pool.insert(i, buf),
                    None => pool.push(buf),
                }
            })
        }
    }
}

impl<T: Send> Clone for BufferPool<T> {
    fn clone(&self) -> BufferPool<T> { BufferPool { pool: self.pool.clone() } }
}

impl<T: Send> Worker<T> {
    /// Pushes data onto the front of this work queue.
    pub fn push(&mut self, t: T) {
        unsafe { (*self.deque.get()).push(t) }
    }
    /// Pops data off the front of the work queue, returning `None` on an empty
    /// queue.
    pub fn pop(&mut self) -> Option<T> {
        unsafe { (*self.deque.get()).pop() }
    }

    /// Gets access to the buffer pool that this worker is attached to. This can
    /// be used to create more deques which share the same buffer pool as this
    /// deque.
    pub fn pool<'a>(&'a mut self) -> &'a mut BufferPool<T> {
        unsafe { &mut (*self.deque.get()).pool }
    }
}

impl<T: Send> Stealer<T> {
    /// Steals work off the end of the queue (opposite of the worker's end)
    pub fn steal(&mut self) -> Stolen<T> {
        unsafe { (*self.deque.get()).steal() }
    }

    /// Gets access to the buffer pool that this stealer is attached to. This
    /// can be used to create more deques which share the same buffer pool as
    /// this deque.
    pub fn pool<'a>(&'a mut self) -> &'a mut BufferPool<T> {
        unsafe { &mut (*self.deque.get()).pool }
    }
}

impl<T: Send> Clone for Stealer<T> {
    fn clone(&self) -> Stealer<T> { Stealer { deque: self.deque.clone() } }
}

// Almost all of this code can be found directly in the paper so I'm not
// personally going to heavily comment what's going on here.

impl<T: Send> Deque<T> {
    fn new(mut pool: BufferPool<T>) -> Deque<T> {
        let buf = pool.alloc(MIN_BITS);
        Deque {
            bottom: AtomicInt::new(0),
            top: AtomicInt::new(0),
            array: AtomicPtr::new(unsafe { cast::transmute(buf) }),
            pool: pool,
        }
    }

    unsafe fn push(&mut self, data: T) {
        let mut b = self.bottom.load(SeqCst);
        let t = self.top.load(SeqCst);
        let mut a = self.array.load(SeqCst);
        let size = b - t;
        if size >= (*a).size() - 1 {
            // You won't find this code in the chase-lev deque paper. This is
            // alluded to in a small footnote, however. We always free a buffer
            // when growing in order to prevent leaks.
            a = self.swap_buffer(b, a, (*a).resize(b, t, 1));
            b = self.bottom.load(SeqCst);
        }
        (*a).put(b, data);
        self.bottom.store(b + 1, SeqCst);
    }

    unsafe fn pop(&mut self) -> Option<T> {
        let b = self.bottom.load(SeqCst);
        let a = self.array.load(SeqCst);
        let b = b - 1;
        self.bottom.store(b, SeqCst);
        let t = self.top.load(SeqCst);
        let size = b - t;
        if size < 0 {
            self.bottom.store(t, SeqCst);
            return None;
        }
        let data = (*a).get(b);
        if size > 0 {
            self.maybe_shrink(b, t);
            return Some(data);
        }
        if self.top.compare_and_swap(t, t + 1, SeqCst) == t {
            self.bottom.store(t + 1, SeqCst);
            return Some(data);
        } else {
            self.bottom.store(t + 1, SeqCst);
            cast::forget(data); // someone else stole this value
            return None;
        }
    }

    unsafe fn steal(&mut self) -> Stolen<T> {
        let t = self.top.load(SeqCst);
        let old = self.array.load(SeqCst);
        let b = self.bottom.load(SeqCst);
        let a = self.array.load(SeqCst);
        let size = b - t;
        if size <= 0 { return Empty }
        if size % (*a).size() == 0 {
            if a == old && t == self.top.load(SeqCst) {
                return Empty
            }
            return Abort
        }
        let data = (*a).get(t);
        if self.top.compare_and_swap(t, t + 1, SeqCst) == t {
            Data(data)
        } else {
            cast::forget(data); // someone else stole this value
            Abort
        }
    }

    unsafe fn maybe_shrink(&mut self, b: int, t: int) {
        let a = self.array.load(SeqCst);
        if b - t < (*a).size() / K && b - t > (1 << MIN_BITS) {
            self.swap_buffer(b, a, (*a).resize(b, t, -1));
        }
    }

    // Helper routine not mentioned in the paper which is used in growing and
    // shrinking buffers to swap in a new buffer into place. As a bit of a
    // recap, the whole point that we need a buffer pool rather than just
    // calling malloc/free directly is that stealers can continue using buffers
    // after this method has called 'free' on it. The continued usage is simply
    // a read followed by a forget, but we must make sure that the memory can
    // continue to be read after we flag this buffer for reclamation.
    unsafe fn swap_buffer(&mut self, b: int, old: *mut Buffer<T>,
                          buf: Buffer<T>) -> *mut Buffer<T> {
        let newbuf: *mut Buffer<T> = cast::transmute(~buf);
        self.array.store(newbuf, SeqCst);
        let ss = (*newbuf).size();
        self.bottom.store(b + ss, SeqCst);
        let t = self.top.load(SeqCst);
        if self.top.compare_and_swap(t, t + ss, SeqCst) != t {
            self.bottom.store(b, SeqCst);
        }
        self.pool.free(cast::transmute(old));
        return newbuf;
    }
}


#[unsafe_destructor]
impl<T: Send> Drop for Deque<T> {
    fn drop(&mut self) {
        let t = self.top.load(SeqCst);
        let b = self.bottom.load(SeqCst);
        let a = self.array.load(SeqCst);
        // Free whatever is leftover in the dequeue, and then move the buffer
        // back into the pool.
        for i in range(t, b) {
            let _: T = unsafe { (*a).get(i) };
        }
        self.pool.free(unsafe { cast::transmute(a) });
    }
}

impl<T: Send> Buffer<T> {
    unsafe fn new(log_size: int) -> Buffer<T> {
        let size = (1 << log_size) * mem::size_of::<T>();
        let buffer = libc::malloc(size as libc::size_t);
        assert!(!buffer.is_null());
        Buffer {
            storage: buffer as *T,
            log_size: log_size,
        }
    }

    fn size(&self) -> int { 1 << self.log_size }

    // Apparently LLVM cannot optimize (foo % (1 << bar)) into this implicitly
    fn mask(&self) -> int { (1 << self.log_size) - 1 }

    // This does not protect against loading duplicate values of the same cell,
    // nor does this clear out the contents contained within. Hence, this is a
    // very unsafe method which the caller needs to treat specially in case a
    // race is lost.
    unsafe fn get(&self, i: int) -> T {
        ptr::read_ptr(self.storage.offset(i & self.mask()))
    }

    // Unsafe because this unsafely overwrites possibly uninitialized or
    // initialized data.
    unsafe fn put(&mut self, i: int, t: T) {
        let ptr = self.storage.offset(i & self.mask());
        ptr::copy_nonoverlapping_memory(ptr as *mut T, &t as *T, 1);
        cast::forget(t);
    }

    // Again, unsafe because this has incredibly dubious ownership violations.
    // It is assumed that this buffer is immediately dropped.
    unsafe fn resize(&self, b: int, t: int, delta: int) -> Buffer<T> {
        let mut buf = Buffer::new(self.log_size + delta);
        for i in range(t, b) {
            buf.put(i, self.get(i));
        }
        return buf;
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for Buffer<T> {
    fn drop(&mut self) {
        // It is assumed that all buffers are empty on drop.
        unsafe { libc::free(self.storage as *mut libc::c_void) }
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::{Data, BufferPool, Abort, Empty, Worker, Stealer};

    use cast;
    use rt::thread::Thread;
    use rand;
    use rand::Rng;
    use sync::atomics::{AtomicBool, INIT_ATOMIC_BOOL, SeqCst,
                        AtomicUint, INIT_ATOMIC_UINT};
    use vec;

    #[test]
    fn smoke() {
        let mut pool = BufferPool::new();
        let (mut w, mut s) = pool.deque();
        assert_eq!(w.pop(), None);
        assert_eq!(s.steal(), Empty);
        w.push(1);
        assert_eq!(w.pop(), Some(1));
        w.push(1);
        assert_eq!(s.steal(), Data(1));
        w.push(1);
        assert_eq!(s.clone().steal(), Data(1));
    }

    #[test]
    fn stealpush() {
        static AMT: int = 100000;
        let mut pool = BufferPool::<int>::new();
        let (mut w, s) = pool.deque();
        let t = do Thread::start {
            let mut s = s;
            let mut left = AMT;
            while left > 0 {
                match s.steal() {
                    Data(i) => {
                        assert_eq!(i, 1);
                        left -= 1;
                    }
                    Abort | Empty => {}
                }
            }
        };

        for _ in range(0, AMT) {
            w.push(1);
        }

        t.join();
    }

    #[test]
    fn stealpush_large() {
        static AMT: int = 100000;
        let mut pool = BufferPool::<(int, int)>::new();
        let (mut w, s) = pool.deque();
        let t = do Thread::start {
            let mut s = s;
            let mut left = AMT;
            while left > 0 {
                match s.steal() {
                    Data((1, 10)) => { left -= 1; }
                    Data(..) => fail!(),
                    Abort | Empty => {}
                }
            }
        };

        for _ in range(0, AMT) {
            w.push((1, 10));
        }

        t.join();
    }

    fn stampede(mut w: Worker<~int>, s: Stealer<~int>,
                nthreads: int, amt: uint) {
        for _ in range(0, amt) {
            w.push(~20);
        }
        let mut remaining = AtomicUint::new(amt);
        let unsafe_remaining: *mut AtomicUint = &mut remaining;

        let threads = range(0, nthreads).map(|_| {
            let s = s.clone();
            do Thread::start {
                unsafe {
                    let mut s = s;
                    while (*unsafe_remaining).load(SeqCst) > 0 {
                        match s.steal() {
                            Data(~20) => {
                                (*unsafe_remaining).fetch_sub(1, SeqCst);
                            }
                            Data(..) => fail!(),
                            Abort | Empty => {}
                        }
                    }
                }
            }
        }).to_owned_vec();

        while remaining.load(SeqCst) > 0 {
            match w.pop() {
                Some(~20) => { remaining.fetch_sub(1, SeqCst); }
                Some(..) => fail!(),
                None => {}
            }
        }

        for thread in threads.move_iter() {
            thread.join();
        }
    }

    #[test]
    fn run_stampede() {
        let mut pool = BufferPool::<~int>::new();
        let (w, s) = pool.deque();
        stampede(w, s, 8, 10000);
    }

    #[test]
    fn many_stampede() {
        static AMT: uint = 4;
        let mut pool = BufferPool::<~int>::new();
        let threads = range(0, AMT).map(|_| {
            let (w, s) = pool.deque();
            do Thread::start {
                stampede(w, s, 4, 10000);
            }
        }).to_owned_vec();

        for thread in threads.move_iter() {
            thread.join();
        }
    }

    #[test]
    fn stress() {
        static AMT: int = 100000;
        static NTHREADS: int = 8;
        static mut DONE: AtomicBool = INIT_ATOMIC_BOOL;
        static mut HITS: AtomicUint = INIT_ATOMIC_UINT;
        let mut pool = BufferPool::<int>::new();
        let (mut w, s) = pool.deque();

        let threads = range(0, NTHREADS).map(|_| {
            let s = s.clone();
            do Thread::start {
                unsafe {
                    let mut s = s;
                    loop {
                        match s.steal() {
                            Data(2) => { HITS.fetch_add(1, SeqCst); }
                            Data(..) => fail!(),
                            _ if DONE.load(SeqCst) => break,
                            _ => {}
                        }
                    }
                }
            }
        }).to_owned_vec();

        let mut rng = rand::task_rng();
        let mut expected = 0;
        while expected < AMT {
            if rng.gen_range(0, 3) == 2 {
                match w.pop() {
                    None => {}
                    Some(2) => unsafe { HITS.fetch_add(1, SeqCst); },
                    Some(_) => fail!(),
                }
            } else {
                expected += 1;
                w.push(2);
            }
        }

        unsafe {
            while HITS.load(SeqCst) < AMT as uint {
                match w.pop() {
                    None => {}
                    Some(2) => { HITS.fetch_add(1, SeqCst); },
                    Some(_) => fail!(),
                }
            }
            DONE.store(true, SeqCst);
        }

        for thread in threads.move_iter() {
            thread.join();
        }

        assert_eq!(unsafe { HITS.load(SeqCst) }, expected as uint);
    }

    #[test]
    #[ignore(cfg(windows))] // apparently windows scheduling is weird?
    fn no_starvation() {
        static AMT: int = 10000;
        static NTHREADS: int = 4;
        static mut DONE: AtomicBool = INIT_ATOMIC_BOOL;
        let mut pool = BufferPool::<(int, uint)>::new();
        let (mut w, s) = pool.deque();

        let (threads, hits) = vec::unzip(range(0, NTHREADS).map(|_| {
            let s = s.clone();
            let unique_box = ~AtomicUint::new(0);
            let thread_box = unsafe {
                *cast::transmute::<&~AtomicUint,**mut AtomicUint>(&unique_box)
            };
            (do Thread::start {
                unsafe {
                    let mut s = s;
                    loop {
                        match s.steal() {
                            Data((1, 2)) => {
                                (*thread_box).fetch_add(1, SeqCst);
                            }
                            Data(..) => fail!(),
                            _ if DONE.load(SeqCst) => break,
                            _ => {}
                        }
                    }
                }
            }, unique_box)
        }));

        let mut rng = rand::task_rng();
        let mut myhit = false;
        let mut iter = 0;
        'outer: loop {
            for _ in range(0, rng.gen_range(0, AMT)) {
                if !myhit && rng.gen_range(0, 3) == 2 {
                    match w.pop() {
                        None => {}
                        Some((1, 2)) => myhit = true,
                        Some(_) => fail!(),
                    }
                } else {
                    w.push((1, 2));
                }
            }
            iter += 1;

            debug!("loop iteration {}", iter);
            for (i, slot) in hits.iter().enumerate() {
                let amt = slot.load(SeqCst);
                debug!("thread {}: {}", i, amt);
                if amt == 0 { continue 'outer; }
            }
            if myhit {
                break
            }
        }

        unsafe { DONE.store(true, SeqCst); }

        for thread in threads.move_iter() {
            thread.join();
        }
    }
}

