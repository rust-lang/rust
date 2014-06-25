// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A concurrent queue used to signal remote event loops
//!
//! This queue implementation is used to send tasks among event loops. This is
//! backed by a multi-producer/single-consumer queue from libstd and uv_async_t
//! handles (to wake up a remote event loop).
//!
//! The uv_async_t is stored next to the event loop, so in order to not keep the
//! event loop alive we use uv_ref and uv_unref in order to control when the
//! async handle is active or not.

#![allow(dead_code)]

use alloc::arc::Arc;
use libc::c_void;
use std::mem;
use std::rt::mutex::NativeMutex;
use std::rt::task::BlockedTask;
use mpsc = std::sync::mpsc_queue;

use async::AsyncWatcher;
use super::{Loop, UvHandle};
use uvll;

enum Message {
    Task(BlockedTask),
    Increment,
    Decrement,
}

struct State {
    handle: *mut uvll::uv_async_t,
    lock: NativeMutex, // see comments in async_cb for why this is needed
    queue: mpsc::Queue<Message>,
}

/// This structure is intended to be stored next to the event loop, and it is
/// used to create new `Queue` structures.
pub struct QueuePool {
    queue: Arc<State>,
    refcnt: uint,
}

/// This type is used to send messages back to the original event loop.
pub struct Queue {
    queue: Arc<State>,
}

extern fn async_cb(handle: *mut uvll::uv_async_t) {
    let pool: &mut QueuePool = unsafe {
        mem::transmute(uvll::get_data_for_uv_handle(handle))
    };
    let state: &State = &*pool.queue;

    // Remember that there is no guarantee about how many times an async
    // callback is called with relation to the number of sends, so process the
    // entire queue in a loop.
    loop {
        match state.queue.pop() {
            mpsc::Data(Task(task)) => {
                let _ = task.wake().map(|t| t.reawaken());
            }
            mpsc::Data(Increment) => unsafe {
                if pool.refcnt == 0 {
                    uvll::uv_ref(state.handle);
                }
                pool.refcnt += 1;
            },
            mpsc::Data(Decrement) => unsafe {
                pool.refcnt -= 1;
                if pool.refcnt == 0 {
                    uvll::uv_unref(state.handle);
                }
            },
            mpsc::Empty | mpsc::Inconsistent => break
        };
    }

    // If the refcount is now zero after processing the queue, then there is no
    // longer a reference on the async handle and it is possible that this event
    // loop can exit. What we're not guaranteed, however, is that a producer in
    // the middle of dropping itself is yet done with the handle. It could be
    // possible that we saw their Decrement message but they have yet to signal
    // on the async handle. If we were to return immediately, the entire uv loop
    // could be destroyed meaning the call to uv_async_send would abort()
    //
    // In order to fix this, an OS mutex is used to wait for the other end to
    // finish before we continue. The drop block on a handle will acquire a
    // mutex and then drop it after both the push and send have been completed.
    // If we acquire the mutex here, then we are guaranteed that there are no
    // longer any senders which are holding on to their handles, so we can
    // safely allow the event loop to exit.
    if pool.refcnt == 0 {
        unsafe {
            let _l = state.lock.lock();
        }
    }
}

impl QueuePool {
    pub fn new(loop_: &mut Loop) -> Box<QueuePool> {
        let handle = UvHandle::alloc(None::<AsyncWatcher>, uvll::UV_ASYNC);
        let state = Arc::new(State {
            handle: handle,
            lock: unsafe {NativeMutex::new()},
            queue: mpsc::Queue::new(),
        });
        let mut q = box QueuePool {
            refcnt: 0,
            queue: state,
        };

        unsafe {
            assert_eq!(uvll::uv_async_init(loop_.handle, handle, async_cb), 0);
            uvll::uv_unref(handle);
            let data = &mut *q as *mut QueuePool as *mut c_void;
            uvll::set_data_for_uv_handle(handle, data);
        }

        return q;
    }

    pub fn queue(&mut self) -> Queue {
        unsafe {
            if self.refcnt == 0 {
                uvll::uv_ref(self.queue.handle);
            }
            self.refcnt += 1;
        }
        Queue { queue: self.queue.clone() }
    }

    pub fn handle(&self) -> *mut uvll::uv_async_t { self.queue.handle }
}

impl Queue {
    pub fn push(&mut self, task: BlockedTask) {
        self.queue.queue.push(Task(task));
        unsafe { uvll::uv_async_send(self.queue.handle); }
    }
}

impl Clone for Queue {
    fn clone(&self) -> Queue {
        // Push a request to increment on the queue, but there's no need to
        // signal the event loop to process it at this time. We're guaranteed
        // that the count is at least one (because we have a queue right here),
        // and if the queue is dropped later on it'll see the increment for the
        // decrement anyway.
        self.queue.queue.push(Increment);
        Queue { queue: self.queue.clone() }
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        // See the comments in the async_cb function for why there is a lock
        // that is acquired only on a drop.
        unsafe {
            let _l = self.queue.lock.lock();
            self.queue.queue.push(Decrement);
            uvll::uv_async_send(self.queue.handle);
        }
    }
}

impl Drop for State {
    fn drop(&mut self) {
        unsafe {
            uvll::uv_close(self.handle, mem::transmute(0));
            // Note that this does *not* free the handle, that is the
            // responsibility of the caller because the uv loop must be closed
            // before we deallocate this uv handle.
        }
    }
}
