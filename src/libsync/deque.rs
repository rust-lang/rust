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
//!     use std::sync::deque::BufferPool;
//!
//!     let mut pool = BufferPool::new();
//!     let (mut worker, mut stealer) = pool.deque();
//!
//!     // Only the worker may push/pop
//!     worker.push(1i);
//!     worker.pop();
//!
//!     // Stealers take data from the other end of the deque
//!     worker.push(1i);
//!     stealer.steal();
//!
//!     // Stealers can be cloned to have many stealers stealing in parallel
//!     worker.push(1i);
//!     let mut stealer2 = stealer.clone();
//!     stealer2.steal();

#![experimental]

// NB: the "buffer pool" strategy is not done for speed, but rather for
//     correctness. For more info, see the comment on `swap_buffer`

// FIXME: all atomic operations in this module use a SeqCst ordering. That is
//      probably overkill

use core::prelude::*;

use alloc::arc::Arc;
use alloc::heap::{allocate, deallocate};
use alloc::boxed::Box;
use collections::{Vec, MutableSeq};
use core::kinds::marker;
use core::mem::{forget, min_align_of, size_of, transmute};
use core::ptr;
use rustrt::exclusive::Exclusive;

use atomic::{AtomicInt, AtomicPtr, SeqCst};

// Once the queue is less than 1/K full, then it will be downsized. Note that
// the deque requires that this number be less than 2.
static K: int = 4;

// Minimum number of bits that a buffer size should be. No buffer will resize to
// under this value, and all deques will initially contain a buffer of this
// size.
//
// The size in question is 1 << MIN_BITS
static MIN_BITS: uint = 7;

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
    deque: Arc<Deque<T>>,
    _noshare: marker::NoSync,
}

/// The stealing half of the work-stealing deque. Stealers have access to the
/// opposite end of the deque from the worker, and they only have access to the
/// `steal` method.
pub struct Stealer<T> {
    deque: Arc<Deque<T>>,
    _noshare: marker::NoSync,
}

/// When stealing some data, this is an enumeration of the possible outcomes.
#[deriving(PartialEq, Show)]
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
    pool: Arc<Exclusive<Vec<Box<Buffer<T>>>>>,
}

/// An internal buffer used by the chase-lev deque. This structure is actually
/// implemented as a circular buffer, and is used as the intermediate storage of
/// the data in the deque.
///
/// This type is implemented with *T instead of Vec<T> for two reasons:
///
///   1. There is nothing safe about using this buffer. This easily allows the
///      same value to be read twice in to rust, and there is nothing to
///      prevent this. The usage by the deque must ensure that one of the
///      values is forgotten. Furthermore, we only ever want to manually run
///      destructors for values in this buffer (on drop) because the bounds
///      are defined by the deque it's owned by.
///
///   2. We can certainly avoid bounds checks using *T instead of Vec<T>, although
///      LLVM is probably pretty good at doing this already.
struct Buffer<T> {
    storage: *const T,
    log_size: uint,
}

impl<T: Send> BufferPool<T> {
    /// Allocates a new buffer pool which in turn can be used to allocate new
    /// deques.
    pub fn new() -> BufferPool<T> {
        BufferPool { pool: Arc::new(Exclusive::new(Vec::new())) }
    }

    /// Allocates a new work-stealing deque which will send/receiving memory to
    /// and from this buffer pool.
    pub fn deque(&self) -> (Worker<T>, Stealer<T>) {
        let a = Arc::new(Deque::new(self.clone()));
        let b = a.clone();
        (Worker { deque: a, _noshare: marker::NoSync },
         Stealer { deque: b, _noshare: marker::NoSync })
    }

    fn alloc(&mut self, bits: uint) -> Box<Buffer<T>> {
        unsafe {
            let mut pool = self.pool.lock();
            match pool.iter().position(|x| x.size() >= (1 << bits)) {
                Some(i) => pool.remove(i).unwrap(),
                None => box Buffer::new(bits)
            }
        }
    }

    fn free(&self, buf: Box<Buffer<T>>) {
        unsafe {
            let mut pool = self.pool.lock();
            match pool.iter().position(|v| v.size() > buf.size()) {
                Some(i) => pool.insert(i, buf),
                None => pool.push(buf),
            }
        }
    }
}

impl<T: Send> Clone for BufferPool<T> {
    fn clone(&self) -> BufferPool<T> { BufferPool { pool: self.pool.clone() } }
}

impl<T: Send> Worker<T> {
    /// Pushes data onto the front of this work queue.
    pub fn push(&self, t: T) {
        unsafe { self.deque.push(t) }
    }
    /// Pops data off the front of the work queue, returning `None` on an empty
    /// queue.
    pub fn pop(&self) -> Option<T> {
        unsafe { self.deque.pop() }
    }

    /// Gets access to the buffer pool that this worker is attached to. This can
    /// be used to create more deques which share the same buffer pool as this
    /// deque.
    pub fn pool<'a>(&'a self) -> &'a BufferPool<T> {
        &self.deque.pool
    }
}

impl<T: Send> Stealer<T> {
    /// Steals work off the end of the queue (opposite of the worker's end)
    pub fn steal(&self) -> Stolen<T> {
        unsafe { self.deque.steal() }
    }

    /// Gets access to the buffer pool that this stealer is attached to. This
    /// can be used to create more deques which share the same buffer pool as
    /// this deque.
    pub fn pool<'a>(&'a self) -> &'a BufferPool<T> {
        &self.deque.pool
    }
}

impl<T: Send> Clone for Stealer<T> {
    fn clone(&self) -> Stealer<T> {
        Stealer { deque: self.deque.clone(), _noshare: marker::NoSync }
    }
}

// Almost all of this code can be found directly in the paper so I'm not
// personally going to heavily comment what's going on here.

impl<T: Send> Deque<T> {
    fn new(mut pool: BufferPool<T>) -> Deque<T> {
        let buf = pool.alloc(MIN_BITS);
        Deque {
            bottom: AtomicInt::new(0),
            top: AtomicInt::new(0),
            array: AtomicPtr::new(unsafe { transmute(buf) }),
            pool: pool,
        }
    }

    unsafe fn push(&self, data: T) {
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

    unsafe fn pop(&self) -> Option<T> {
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
            forget(data); // someone else stole this value
            return None;
        }
    }

    unsafe fn steal(&self) -> Stolen<T> {
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
            forget(data); // someone else stole this value
            Abort
        }
    }

    unsafe fn maybe_shrink(&self, b: int, t: int) {
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
    unsafe fn swap_buffer(&self, b: int, old: *mut Buffer<T>,
                          buf: Buffer<T>) -> *mut Buffer<T> {
        let newbuf: *mut Buffer<T> = transmute(box buf);
        self.array.store(newbuf, SeqCst);
        let ss = (*newbuf).size();
        self.bottom.store(b + ss, SeqCst);
        let t = self.top.load(SeqCst);
        if self.top.compare_and_swap(t, t + ss, SeqCst) != t {
            self.bottom.store(b, SeqCst);
        }
        self.pool.free(transmute(old));
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
        self.pool.free(unsafe { transmute(a) });
    }
}

#[inline]
fn buffer_alloc_size<T>(log_size: uint) -> uint {
    (1 << log_size) * size_of::<T>()
}

impl<T: Send> Buffer<T> {
    unsafe fn new(log_size: uint) -> Buffer<T> {
        let size = buffer_alloc_size::<T>(log_size);
        let buffer = allocate(size, min_align_of::<T>());
        Buffer {
            storage: buffer as *const T,
            log_size: log_size,
        }
    }

    fn size(&self) -> int { 1 << self.log_size }

    // Apparently LLVM cannot optimize (foo % (1 << bar)) into this implicitly
    fn mask(&self) -> int { (1 << self.log_size) - 1 }

    unsafe fn elem(&self, i: int) -> *const T {
        self.storage.offset(i & self.mask())
    }

    // This does not protect against loading duplicate values of the same cell,
    // nor does this clear out the contents contained within. Hence, this is a
    // very unsafe method which the caller needs to treat specially in case a
    // race is lost.
    unsafe fn get(&self, i: int) -> T {
        ptr::read(self.elem(i))
    }

    // Unsafe because this unsafely overwrites possibly uninitialized or
    // initialized data.
    unsafe fn put(&self, i: int, t: T) {
        ptr::write(self.elem(i) as *mut T, t);
    }

    // Again, unsafe because this has incredibly dubious ownership violations.
    // It is assumed that this buffer is immediately dropped.
    unsafe fn resize(&self, b: int, t: int, delta: int) -> Buffer<T> {
        // NB: not entirely obvious, but thanks to 2's complement,
        // casting delta to uint and then adding gives the desired
        // effect.
        let buf = Buffer::new(self.log_size + delta as uint);
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
        let size = buffer_alloc_size::<T>(self.log_size);
        unsafe { deallocate(self.storage as *mut u8, size, min_align_of::<T>()) }
    }
}

#[cfg(test)]
mod tests {
    use std::prelude::*;
    use super::{Data, BufferPool, Abort, Empty, Worker, Stealer};

    use std::mem;
    use std::rt::thread::Thread;
    use std::rand;
    use std::rand::Rng;
    use atomic::{AtomicBool, INIT_ATOMIC_BOOL, SeqCst,
                  AtomicUint, INIT_ATOMIC_UINT};
    use std::vec;

    #[test]
    fn smoke() {
        let pool = BufferPool::new();
        let (w, s) = pool.deque();
        assert_eq!(w.pop(), None);
        assert_eq!(s.steal(), Empty);
        w.push(1i);
        assert_eq!(w.pop(), Some(1));
        w.push(1);
        assert_eq!(s.steal(), Data(1));
        w.push(1);
        assert_eq!(s.clone().steal(), Data(1));
    }

    #[test]
    fn stealpush() {
        static AMT: int = 100000;
        let pool = BufferPool::<int>::new();
        let (w, s) = pool.deque();
        let t = Thread::start(proc() {
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
        });

        for _ in range(0, AMT) {
            w.push(1);
        }

        t.join();
    }

    #[test]
    fn stealpush_large() {
        static AMT: int = 100000;
        let pool = BufferPool::<(int, int)>::new();
        let (w, s) = pool.deque();
        let t = Thread::start(proc() {
            let mut left = AMT;
            while left > 0 {
                match s.steal() {
                    Data((1, 10)) => { left -= 1; }
                    Data(..) => fail!(),
                    Abort | Empty => {}
                }
            }
        });

        for _ in range(0, AMT) {
            w.push((1, 10));
        }

        t.join();
    }

    fn stampede(w: Worker<Box<int>>, s: Stealer<Box<int>>,
                nthreads: int, amt: uint) {
        for _ in range(0, amt) {
            w.push(box 20);
        }
        let mut remaining = AtomicUint::new(amt);
        let unsafe_remaining: *mut AtomicUint = &mut remaining;

        let threads = range(0, nthreads).map(|_| {
            let s = s.clone();
            Thread::start(proc() {
                unsafe {
                    while (*unsafe_remaining).load(SeqCst) > 0 {
                        match s.steal() {
                            Data(box 20) => {
                                (*unsafe_remaining).fetch_sub(1, SeqCst);
                            }
                            Data(..) => fail!(),
                            Abort | Empty => {}
                        }
                    }
                }
            })
        }).collect::<Vec<Thread<()>>>();

        while remaining.load(SeqCst) > 0 {
            match w.pop() {
                Some(box 20) => { remaining.fetch_sub(1, SeqCst); }
                Some(..) => fail!(),
                None => {}
            }
        }

        for thread in threads.into_iter() {
            thread.join();
        }
    }

    #[test]
    fn run_stampede() {
        let pool = BufferPool::<Box<int>>::new();
        let (w, s) = pool.deque();
        stampede(w, s, 8, 10000);
    }

    #[test]
    fn many_stampede() {
        static AMT: uint = 4;
        let pool = BufferPool::<Box<int>>::new();
        let threads = range(0, AMT).map(|_| {
            let (w, s) = pool.deque();
            Thread::start(proc() {
                stampede(w, s, 4, 10000);
            })
        }).collect::<Vec<Thread<()>>>();

        for thread in threads.into_iter() {
            thread.join();
        }
    }

    #[test]
    fn stress() {
        static AMT: int = 100000;
        static NTHREADS: int = 8;
        static mut DONE: AtomicBool = INIT_ATOMIC_BOOL;
        static mut HITS: AtomicUint = INIT_ATOMIC_UINT;
        let pool = BufferPool::<int>::new();
        let (w, s) = pool.deque();

        let threads = range(0, NTHREADS).map(|_| {
            let s = s.clone();
            Thread::start(proc() {
                unsafe {
                    loop {
                        match s.steal() {
                            Data(2) => { HITS.fetch_add(1, SeqCst); }
                            Data(..) => fail!(),
                            _ if DONE.load(SeqCst) => break,
                            _ => {}
                        }
                    }
                }
            })
        }).collect::<Vec<Thread<()>>>();

        let mut rng = rand::task_rng();
        let mut expected = 0;
        while expected < AMT {
            if rng.gen_range(0i, 3) == 2 {
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

        for thread in threads.into_iter() {
            thread.join();
        }

        assert_eq!(unsafe { HITS.load(SeqCst) }, expected as uint);
    }

    #[test]
    #[cfg_attr(windows, ignore)] // apparently windows scheduling is weird?
    fn no_starvation() {
        static AMT: int = 10000;
        static NTHREADS: int = 4;
        static mut DONE: AtomicBool = INIT_ATOMIC_BOOL;
        let pool = BufferPool::<(int, uint)>::new();
        let (w, s) = pool.deque();

        let (threads, hits) = vec::unzip(range(0, NTHREADS).map(|_| {
            let s = s.clone();
            let unique_box = box AtomicUint::new(0);
            let thread_box = unsafe {
                *mem::transmute::<&Box<AtomicUint>,
                                  *const *mut AtomicUint>(&unique_box)
            };
            (Thread::start(proc() {
                unsafe {
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
            }), unique_box)
        }));

        let mut rng = rand::task_rng();
        let mut myhit = false;
        'outer: loop {
            for _ in range(0, rng.gen_range(0, AMT)) {
                if !myhit && rng.gen_range(0i, 3) == 2 {
                    match w.pop() {
                        None => {}
                        Some((1, 2)) => myhit = true,
                        Some(_) => fail!(),
                    }
                } else {
                    w.push((1, 2));
                }
            }

            for slot in hits.iter() {
                let amt = slot.load(SeqCst);
                if amt == 0 { continue 'outer; }
            }
            if myhit {
                break
            }
        }

        unsafe { DONE.store(true, SeqCst); }

        for thread in threads.into_iter() {
            thread.join();
        }
    }
}
