// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use vec;
use unstable::atomics::{atomic_store, atomic_load, AtomicUint, fence, SeqCst, Acquire, Release, Relaxed};
use unstable::sync::{UnsafeArc, LittleLock};
use cast;
use option::{Option, Some, None};
use iter::range;
use clone::Clone;
use kinds::Send;

pub struct WorkQueue<T> {
    priv state: UnsafeArc<State<T>>,
}

impl<T: Send> WorkQueue<T> {
    pub fn new() -> WorkQueue<T> {
        WorkQueue::with_capacity(2)
    }

    pub fn with_capacity(capacity: uint) -> WorkQueue<T> {
        WorkQueue{
            state: UnsafeArc::new(State::with_capacity(capacity))
        }
    }

    pub fn push(&mut self, value: T) {
        unsafe { (*self.state.get()).push(value) }
    }

    pub fn pop(&mut self) -> Option<T> {
        unsafe { (*self.state.get()).pop() }
    }

    pub fn steal(&mut self) -> Option<T> {
        unsafe { (*self.state.get()).steal() }
    }

    pub fn is_empty(&mut self) -> bool {
        unsafe { (*self.state.get()).is_empty() }
    }

    pub fn len(&mut self) -> uint {
        unsafe { (*self.state.get()).len() }
    }
}

impl<T: Send> Clone for WorkQueue<T> {
    fn clone(&self) -> WorkQueue<T> {
        WorkQueue {
            state: self.state.clone()
        }
    }
}

struct State<T> {
    array: ~[*mut T],
    mask: uint,
    headIndex: AtomicUint,
    tailIndex: AtomicUint,
    lock: LittleLock,
}

impl<T: Send> State<T> {
    fn with_capacity(size: uint) -> State<T> {
        let mut state = State{
            array: vec::with_capacity(size),
            mask: size-1,
            headIndex: AtomicUint::new(0),
            tailIndex: AtomicUint::new(0),
            lock: LittleLock::new()
        };
        unsafe {
            vec::raw::set_len(&mut state.array, size);
        }
        state
    }

    fn push(&mut self, value: T) {
        let mut tail = self.tailIndex.load(Acquire);
        if tail < self.headIndex.load(Acquire) + self.mask {
            unsafe {
                atomic_store(&mut self.array[tail & self.mask], cast::transmute(value), Relaxed);
            }
            self.tailIndex.store(tail+1, Release);
        } else {
            unsafe {
                let value: *mut T =  cast::transmute(value);
                self.lock.lock(|| {
                    let head = self.headIndex.load(Acquire);
                    let count = self.len();
                    if count >= self.mask {
                        let arraySize = self.array.len();
                        let mask = self.mask;
                        let mut newArray = vec::with_capacity(arraySize*2);
                        vec::raw::set_len(&mut newArray, arraySize*2);
                        for i in range(0, count) {
                            newArray[i] = self.array[(i+head) & mask];
                        }
                        self.array = newArray;
                        self.headIndex.store(0, Release);
                        self.tailIndex.store(count, Release);
                        tail = count;
                        self.mask = (mask * 2) | 1;
                    }
                    atomic_store(&mut self.array[tail & self.mask], value, Relaxed);
                    self.tailIndex.store(tail+1, Release);
                });
            }
        }
    }

    fn pop(&mut self) -> Option<T> {
        let mut tail = self.tailIndex.load(Acquire);
        if tail == 0 {
            return None
        }
        tail -= 1;
        self.tailIndex.store(tail, Release);
        fence(SeqCst);
        unsafe {
            if self.headIndex.load(Acquire) <= tail {
                Some(cast::transmute(atomic_load(&mut self.array[tail & self.mask], Relaxed)))
            } else {
                self.lock.lock(|| {
                    if self.headIndex.load(Acquire) <= tail {
                        Some(cast::transmute(atomic_load(&mut self.array[tail & self.mask], Relaxed)))
                    } else {
                        self.tailIndex.store(tail+1, Release);
                        None
                    }
                })
            }
        }
    }

    fn steal(&mut self) -> Option<T> {
        unsafe {
            match self.lock.try_lock(|| {
                let head = self.headIndex.load(Acquire);
                self.headIndex.store(head+1, Release);
                fence(SeqCst);
                if head < self.tailIndex.load(Acquire) {
                    Some(cast::transmute(atomic_load(&mut self.array[head & self.mask], Relaxed)))
                } else {
                    self.headIndex.store(head, Release);
                    None
                }
            }) {
                Some(T) => T,
                None => None
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.headIndex.load(Acquire) >= self.tailIndex.load(Acquire)
    }

    fn len(&self) -> uint {
        self.tailIndex.load(Acquire) - self.headIndex.load(Acquire)
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use task;
    use comm;
    use unstable::sync::{UnsafeArc};
    use unstable::atomics::{AtomicUint, Relaxed};
    use super::WorkQueue;

    #[test]
    fn test() {
        let mut q = WorkQueue::with_capacity(10);
        q.push(1);
        assert_eq!(Some(1), q.pop());
        assert_eq!(None, q.steal());
        q.push(2);
        assert_eq!(Some(2), q.steal());
    }

    #[test]
    fn test_grow() {
        let mut q = WorkQueue::with_capacity(2);
        q.push(1);
        assert_eq!(Some(1), q.pop());
        assert_eq!(None, q.steal());
        q.push(2);
        assert_eq!(Some(2), q.steal());
        q.push(3);
        q.push(4);
        assert_eq!(Some(4), q.pop());
        assert_eq!(Some(3), q.pop());
        assert_eq!(None, q.steal());
    }

    #[test]
    fn test_steal() {
        let work_units = 1000u;
        let stealers = 8u;
        let q = WorkQueue::with_capacity(100);
        let counter = UnsafeArc::new(AtomicUint::new(0));
        let mut completion_ports = ~[];

        let (port, chan)  = comm::stream();
        let (completion_port, completion_chan) = comm::stream();
        completion_ports.push(completion_port);
        chan.send(q.clone());
        {
            let counter = counter.clone();
            do task::spawn_sched(task::SingleThreaded) {
                let mut q = port.recv();
                for i in range(0, work_units) {
                    q.push(i);
                }

                let mut count = 0u;
                loop {
                    match q.pop() {
                        Some(_) => unsafe {
                            count += 1;
                            (*counter.get()).fetch_add(1, Relaxed);
                            // simulate work
                            task::deschedule();
                        },
                        None => break,
                    }
                }
                debug!("count: {}", count);
                completion_chan.send(0);
            }
        }

        for _ in range(0, stealers) {
            let (port, chan)  = comm::stream();
            let (completion_port, completion_chan) = comm::stream();
            completion_ports.push(completion_port);
            chan.send(q.clone());
            let counter = counter.clone();
            do task::spawn_sched(task::SingleThreaded) {
                let mut count = 0u;
                let mut q = port.recv();
                loop {
                    match q.steal() {
                        Some(_) => unsafe {
                            count += 1;
                            (*counter.get()).fetch_add(1, Relaxed);
                        },
                        None => (),
                    }
                    // simulate work
                    task::deschedule();
                    unsafe {
                        if (*counter.get()).load(Relaxed) == work_units {
                            break
                        }
                    }
                }
                debug!("count: {}", count);
                completion_chan.send(0);
            }
        }

        // wait for all tasks to finish work
        for completion_port in completion_ports.iter() {
            assert_eq!(0, completion_port.recv());
        }

        unsafe {
            assert_eq!(work_units, (*counter.get()).load(Relaxed));
        }
    }
}
