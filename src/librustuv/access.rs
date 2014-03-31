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

use std::cast;
use std::sync::arc::UnsafeArc;
use std::rt::task::{BlockedTask, Task};
use std::rt::local::Local;

use homing::HomingMissile;

pub struct Access {
    inner: UnsafeArc<Inner>,
}

pub struct Guard<'a> {
    access: &'a mut Access,
    missile: Option<HomingMissile>,
}

struct Inner {
    queue: ~[BlockedTask],
    held: bool,
}

impl Access {
    pub fn new() -> Access {
        Access {
            inner: UnsafeArc::new(Inner {
                queue: ~[],
                held: false,
            })
        }
    }

    pub fn grant<'a>(&'a mut self, missile: HomingMissile) -> Guard<'a> {
        // This unsafety is actually OK because the homing missile argument
        // guarantees that we're on the same event loop as all the other objects
        // attempting to get access granted.
        let inner: &mut Inner = unsafe { cast::transmute(self.inner.get()) };

        if inner.held {
            let t: ~Task = Local::take();
            t.deschedule(1, |task| {
                inner.queue.push(task);
                Ok(())
            });
            assert!(inner.held);
        } else {
            inner.held = true;
        }

        Guard { access: self, missile: Some(missile) }
    }
}

impl Clone for Access {
    fn clone(&self) -> Access {
        Access { inner: self.inner.clone() }
    }
}

#[unsafe_destructor]
impl<'a> Drop for Guard<'a> {
    fn drop(&mut self) {
        // This guard's homing missile is still armed, so we're guaranteed to be
        // on the same I/O event loop, so this unsafety should be ok.
        assert!(self.missile.is_some());
        let inner: &mut Inner = unsafe {
            cast::transmute(self.access.inner.get())
        };

        match inner.queue.shift() {
            // Here we have found a task that was waiting for access, and we
            // current have the "access lock" we need to relinquish access to
            // this sleeping task.
            //
            // To do so, we first drop out homing missile and we then reawaken
            // the task. In reawakening the task, it will be immediately
            // scheduled on this scheduler. Because we might be woken up on some
            // other scheduler, we drop our homing missile before we reawaken
            // the task.
            Some(task) => {
                drop(self.missile.take());
                let _ = task.wake().map(|t| t.reawaken());
            }
            None => { inner.held = false; }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        assert!(!self.held);
        assert_eq!(self.queue.len(), 0);
    }
}
