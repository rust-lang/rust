// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// An exclusive access primitive
///
/// This primitive is used to gain exclusive access to read() and write() in uv.
/// It is assumed that all invocations of this struct happen on the same thread
/// (the uv event loop).

use alloc::arc::Arc;
use std::mem;
use std::rt::local::Local;
use std::rt::task::{BlockedTask, Task};
use std::cell::UnsafeCell;

use homing::HomingMissile;

pub struct Access<T> {
    inner: Arc<UnsafeCell<Inner<T>>>,
}

pub struct Guard<'a, T:'static> {
    access: &'a mut Access<T>,
    missile: Option<HomingMissile>,
}

struct Inner<T> {
    queue: Vec<(BlockedTask, uint)>,
    held: bool,
    closed: bool,
    data: T,
}

impl<T: Send> Access<T> {
    pub fn new(data: T) -> Access<T> {
        Access {
            inner: Arc::new(UnsafeCell::new(Inner {
                queue: vec![],
                held: false,
                closed: false,
                data: data,
            }))
        }
    }

    pub fn grant<'a>(&'a mut self, token: uint,
                     missile: HomingMissile) -> Guard<'a, T> {
        // This unsafety is actually OK because the homing missile argument
        // guarantees that we're on the same event loop as all the other objects
        // attempting to get access granted.
        let inner = unsafe { &mut *self.inner.get() };

        if inner.held {
            let t: Box<Task> = Local::take();
            t.deschedule(1, |task| {
                inner.queue.push((task, token));
                Ok(())
            });
            assert!(inner.held);
        } else {
            inner.held = true;
        }

        Guard { access: self, missile: Some(missile) }
    }

    pub fn unsafe_get(&self) -> *mut T {
        unsafe { &mut (*self.inner.get()).data  as *mut _ }
    }

    // Safe version which requires proof that you are on the home scheduler.
    pub fn get_mut<'a>(&'a mut self, _missile: &HomingMissile) -> &'a mut T {
        unsafe { &mut *self.unsafe_get() }
    }

    pub fn close(&self, _missile: &HomingMissile) {
        // This unsafety is OK because with a homing missile we're guaranteed to
        // be the only task looking at the `closed` flag (and are therefore
        // allowed to modify it). Additionally, no atomics are necessary because
        // everyone's running on the same thread and has already done the
        // necessary synchronization to be running on this thread.
        unsafe { (*self.inner.get()).closed = true; }
    }

    // Dequeue a blocked task with a specified token. This is unsafe because it
    // is only safe to invoke while on the home event loop, and there is no
    // guarantee that this i being invoked on the home event loop.
    pub unsafe fn dequeue(&mut self, token: uint) -> Option<BlockedTask> {
        let inner = &mut *self.inner.get();
        match inner.queue.iter().position(|&(_, t)| t == token) {
            Some(i) => Some(inner.queue.remove(i).unwrap().val0()),
            None => None,
        }
    }

    /// Test whether this access is closed, using a homing missile to prove
    /// that it's safe
    pub fn is_closed(&self, _missile: &HomingMissile) -> bool {
        unsafe { (*self.inner.get()).closed }
    }
}

impl<T: Send> Clone for Access<T> {
    fn clone(&self) -> Access<T> {
        Access { inner: self.inner.clone() }
    }
}

impl<'a, T: Send> Guard<'a, T> {
    pub fn is_closed(&self) -> bool {
        // See above for why this unsafety is ok, it just applies to the read
        // instead of the write.
        unsafe { (*self.access.inner.get()).closed }
    }
}

impl<'a, T: Send> Deref<T> for Guard<'a, T> {
    fn deref<'a>(&'a self) -> &'a T {
        // A guard represents exclusive access to a piece of data, so it's safe
        // to hand out shared and mutable references
        unsafe { &(*self.access.inner.get()).data }
    }
}

impl<'a, T: Send> DerefMut<T> for Guard<'a, T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe { &mut (*self.access.inner.get()).data }
    }
}

#[unsafe_destructor]
impl<'a, T> Drop for Guard<'a, T> {
    fn drop(&mut self) {
        // This guard's homing missile is still armed, so we're guaranteed to be
        // on the same I/O event loop, so this unsafety should be ok.
        assert!(self.missile.is_some());
        let inner: &mut Inner<T> = unsafe {
            mem::transmute(self.access.inner.get())
        };

        match inner.queue.remove(0) {
            // Here we have found a task that was waiting for access, and we
            // current have the "access lock" we need to relinquish access to
            // this sleeping task.
            //
            // To do so, we first drop out homing missile and we then reawaken
            // the task. In reawakening the task, it will be immediately
            // scheduled on this scheduler. Because we might be woken up on some
            // other scheduler, we drop our homing missile before we reawaken
            // the task.
            Some((task, _)) => {
                drop(self.missile.take());
                task.reawaken();
            }
            None => { inner.held = false; }
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        assert!(!self.held);
        assert_eq!(self.queue.len(), 0);
    }
}
