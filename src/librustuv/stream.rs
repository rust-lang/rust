// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cast;
use std::libc::{c_int, size_t, ssize_t, c_void};
use std::ptr;
use std::rt::BlockedTask;
use std::rt::local::Local;
use std::rt::sched::Scheduler;

use super::{UvError, Buf, slice_to_uv_buf, Request};
use uvll;

// This is a helper structure which is intended to get embedded into other
// Watcher structures. This structure will retain a handle to the underlying
// uv_stream_t instance, and all I/O operations assume that it's already located
// on the appropriate scheduler.
pub struct StreamWatcher {
    handle: *uvll::uv_stream_t,

    // Cache the last used uv_write_t so we don't have to allocate a new one on
    // every call to uv_write(). Ideally this would be a stack-allocated
    // structure, but currently we don't have mappings for all the structures
    // defined in libuv, so we're foced to malloc this.
    priv last_write_req: Option<Request>,
}

struct ReadContext {
    buf: Option<Buf>,
    result: ssize_t,
    task: Option<BlockedTask>,
}

struct WriteContext {
    result: c_int,
    task: Option<BlockedTask>,
}

impl StreamWatcher {
    // Creates a new helper structure which should be then embedded into another
    // watcher. This provides the generic read/write methods on streams.
    //
    // This structure will *not* close the stream when it is dropped. It is up
    // to the enclosure structure to be sure to call the close method (which
    // will block the task). Note that this is also required to prevent memory
    // leaks.
    //
    // It should also be noted that the `data` field of the underlying uv handle
    // will be manipulated on each of the methods called on this watcher.
    // Wrappers should ensure to always reset the field to an appropriate value
    // if they rely on the field to perform an action.
    pub fn new(stream: *uvll::uv_stream_t) -> StreamWatcher {
        StreamWatcher {
            handle: stream,
            last_write_req: None,
        }
    }

    pub fn read(&mut self, buf: &mut [u8]) -> Result<uint, UvError> {
        // Send off the read request, but don't block until we're sure that the
        // read request is queued.
        match unsafe {
            uvll::uv_read_start(self.handle, alloc_cb, read_cb)
        } {
            0 => {
                let mut rcx = ReadContext {
                    buf: Some(slice_to_uv_buf(buf)),
                    result: 0,
                    task: None,
                };
                unsafe {
                    uvll::set_data_for_uv_handle(self.handle, &rcx)
                }
                let scheduler: ~Scheduler = Local::take();
                do scheduler.deschedule_running_task_and_then |_sched, task| {
                    rcx.task = Some(task);
                }
                match rcx.result {
                    n if n < 0 => Err(UvError(n as c_int)),
                    n => Ok(n as uint),
                }
            }
            n => Err(UvError(n))
        }
    }

    pub fn write(&mut self, buf: &[u8]) -> Result<(), UvError> {
        // Prepare the write request, either using a cached one or allocating a
        // new one
        if self.last_write_req.is_none() {
            self.last_write_req = Some(Request::new(uvll::UV_WRITE));
        }
        let req = self.last_write_req.get_ref();

        // Send off the request, but be careful to not block until we're sure
        // that the write reqeust is queued. If the reqeust couldn't be queued,
        // then we should return immediately with an error.
        match unsafe {
            uvll::uv_write(req.handle, self.handle, [slice_to_uv_buf(buf)],
                           write_cb)
        } {
            0 => {
                let mut wcx = WriteContext { result: 0, task: None, };
                req.set_data(&wcx);
                let scheduler: ~Scheduler = Local::take();
                do scheduler.deschedule_running_task_and_then |_sched, task| {
                    wcx.task = Some(task);
                }
                match wcx.result {
                    0 => Ok(()),
                    n => Err(UvError(n)),
                }
            }
            n => Err(UvError(n)),
        }
    }

    // This will deallocate an internally used memory, along with closing the
    // handle (and freeing it).
    //
    // The `synchronous` flag dictates whether this handle is closed
    // synchronously (the task is blocked) or asynchronously (the task is not
    // block, but the handle is still deallocated).
    pub fn close(&mut self, synchronous: bool) {
        if synchronous {
            let mut closing_task = None;
            unsafe {
                uvll::set_data_for_uv_handle(self.handle, &closing_task);
            }

            // Wait for this stream to close because it possibly represents a remote
            // connection which may have consequences if we close asynchronously.
            let sched: ~Scheduler = Local::take();
            do sched.deschedule_running_task_and_then |_, task| {
                closing_task = Some(task);
                unsafe { uvll::uv_close(self.handle, close_cb) }
            }
        } else {
            unsafe {
                uvll::set_data_for_uv_handle(self.handle, ptr::null::<u8>());
                uvll::uv_close(self.handle, close_cb)
            }
        }

        extern fn close_cb(handle: *uvll::uv_handle_t) {
            let data: *c_void = unsafe { uvll::get_data_for_uv_handle(handle) };
            unsafe { uvll::free_handle(handle) }
            if data.is_null() { return }

            let closing_task: &mut Option<BlockedTask> = unsafe {
                cast::transmute(data)
            };
            let task = closing_task.take_unwrap();
            let scheduler: ~Scheduler = Local::take();
            scheduler.resume_blocked_task_immediately(task);
        }
    }
}

// This allocation callback expects to be invoked once and only once. It will
// unwrap the buffer in the ReadContext stored in the stream and return it. This
// will fail if it is called more than once.
extern fn alloc_cb(stream: *uvll::uv_stream_t, _hint: size_t) -> Buf {
    let rcx: &mut ReadContext = unsafe {
        cast::transmute(uvll::get_data_for_uv_handle(stream))
    };
    rcx.buf.take().expect("alloc_cb called more than once")
}

// When a stream has read some data, we will always forcibly stop reading and
// return all the data read (even if it didn't fill the whole buffer).
extern fn read_cb(handle: *uvll::uv_stream_t, nread: ssize_t, _buf: Buf) {
    let rcx: &mut ReadContext = unsafe {
        cast::transmute(uvll::get_data_for_uv_handle(handle))
    };
    // Stop reading so that no read callbacks are
    // triggered before the user calls `read` again.
    // XXX: Is there a performance impact to calling
    // stop here?
    unsafe { assert_eq!(uvll::uv_read_stop(handle), 0); }
    rcx.result = nread;

    let scheduler: ~Scheduler = Local::take();
    scheduler.resume_blocked_task_immediately(rcx.task.take_unwrap());
}

// Unlike reading, the WriteContext is stored in the uv_write_t request. Like
// reading, however, all this does is wake up the blocked task after squirreling
// away the error code as a result.
extern fn write_cb(req: *uvll::uv_write_t, status: c_int) {
    if status == uvll::ECANCELED { return }
    // Remember to not free the request because it is re-used between writes on
    // the same stream.
    let req = Request::wrap(req);
    let wcx: &mut WriteContext = unsafe { cast::transmute(req.get_data()) };
    wcx.result = status;
    req.defuse();

    let sched: ~Scheduler = Local::take();
    sched.resume_blocked_task_immediately(wcx.task.take_unwrap());
}
