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
use libc::{c_int, size_t, ssize_t};
use std::ptr;
use std::rt::task::BlockedTask;

use Loop;
use super::{UvError, Buf, slice_to_uv_buf, Request, wait_until_woken_after,
            ForbidUnwind, wakeup};
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
        // This read operation needs to get canceled on an unwind via libuv's
        // uv_read_stop function
        let _f = ForbidUnwind::new("stream read");

        let mut rcx = ReadContext {
            buf: Some(slice_to_uv_buf(buf)),
            result: 0,
            task: None,
        };
        // When reading a TTY stream on windows, libuv will invoke alloc_cb
        // immediately as part of the call to alloc_cb. What this means is that
        // we must be ready for this to happen (by setting the data in the uv
        // handle). In theory this otherwise doesn't need to happen until after
        // the read is succesfully started.
        unsafe {
            uvll::set_data_for_uv_handle(self.handle, &rcx)
        }

        // Send off the read request, but don't block until we're sure that the
        // read request is queued.
        match unsafe {
            uvll::uv_read_start(self.handle, alloc_cb, read_cb)
        } {
            0 => {
                let loop_ = unsafe { uvll::get_loop_for_uv_handle(self.handle) };
                wait_until_woken_after(&mut rcx.task, &Loop::wrap(loop_), || {});
                match rcx.result {
                    n if n < 0 => Err(UvError(n as c_int)),
                    n => Ok(n as uint),
                }
            }
            n => Err(UvError(n))
        }
    }

    pub fn write(&mut self, buf: &[u8]) -> Result<(), UvError> {
        // The ownership of the write request is dubious if this function
        // unwinds. I believe that if the write_cb fails to re-schedule the task
        // then the write request will be leaked.
        let _f = ForbidUnwind::new("stream write");

        // Prepare the write request, either using a cached one or allocating a
        // new one
        let mut req = match self.last_write_req.take() {
            Some(req) => req, None => Request::new(uvll::UV_WRITE),
        };
        req.set_data(ptr::null::<()>());

        // Send off the request, but be careful to not block until we're sure
        // that the write reqeust is queued. If the reqeust couldn't be queued,
        // then we should return immediately with an error.
        match unsafe {
            uvll::uv_write(req.handle, self.handle, [slice_to_uv_buf(buf)],
                           write_cb)
        } {
            0 => {
                let mut wcx = WriteContext { result: 0, task: None, };
                req.defuse(); // uv callback now owns this request

                let loop_ = unsafe { uvll::get_loop_for_uv_handle(self.handle) };
                wait_until_woken_after(&mut wcx.task, &Loop::wrap(loop_), || {
                    req.set_data(&wcx);
                });
                self.last_write_req = Some(Request::wrap(req.handle));
                match wcx.result {
                    0 => Ok(()),
                    n => Err(UvError(n)),
                }
            }
            n => Err(UvError(n)),
        }
    }
}

// This allocation callback expects to be invoked once and only once. It will
// unwrap the buffer in the ReadContext stored in the stream and return it. This
// will fail if it is called more than once.
extern fn alloc_cb(stream: *uvll::uv_stream_t, _hint: size_t, buf: *mut Buf) {
    uvdebug!("alloc_cb");
    unsafe {
        let rcx: &mut ReadContext =
            cast::transmute(uvll::get_data_for_uv_handle(stream));
        *buf = rcx.buf.take().expect("stream alloc_cb called more than once");
    }
}

// When a stream has read some data, we will always forcibly stop reading and
// return all the data read (even if it didn't fill the whole buffer).
extern fn read_cb(handle: *uvll::uv_stream_t, nread: ssize_t, _buf: *Buf) {
    uvdebug!("read_cb {}", nread);
    assert!(nread != uvll::ECANCELED as ssize_t);
    let rcx: &mut ReadContext = unsafe {
        cast::transmute(uvll::get_data_for_uv_handle(handle))
    };
    // Stop reading so that no read callbacks are
    // triggered before the user calls `read` again.
    // FIXME: Is there a performance impact to calling
    // stop here?
    unsafe { assert_eq!(uvll::uv_read_stop(handle), 0); }
    rcx.result = nread;

    wakeup(&mut rcx.task);
}

// Unlike reading, the WriteContext is stored in the uv_write_t request. Like
// reading, however, all this does is wake up the blocked task after squirreling
// away the error code as a result.
extern fn write_cb(req: *uvll::uv_write_t, status: c_int) {
    let mut req = Request::wrap(req);
    assert!(status != uvll::ECANCELED);
    // Remember to not free the request because it is re-used between writes on
    // the same stream.
    let wcx: &mut WriteContext = unsafe { req.get_data() };
    wcx.result = status;
    req.defuse();

    wakeup(&mut wcx.task);
}
