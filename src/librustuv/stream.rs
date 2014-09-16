// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::{c_int, size_t, ssize_t};
use std::mem;
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
    pub handle: *mut uvll::uv_stream_t,

    // Cache the last used uv_write_t so we don't have to allocate a new one on
    // every call to uv_write(). Ideally this would be a stack-allocated
    // structure, but currently we don't have mappings for all the structures
    // defined in libuv, so we're forced to malloc this.
    last_write_req: Option<Request>,

    blocked_writer: Option<BlockedTask>,
}

struct ReadContext {
    buf: Option<Buf>,
    result: ssize_t,
    task: Option<BlockedTask>,
}

struct WriteContext {
    result: c_int,
    stream: *mut StreamWatcher,
    data: Option<Vec<u8>>,
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
    pub fn new(stream: *mut uvll::uv_stream_t,
               init: bool) -> StreamWatcher {
        if init {
            unsafe { uvll::set_data_for_uv_handle(stream, 0 as *mut int) }
        }
        StreamWatcher {
            handle: stream,
            last_write_req: None,
            blocked_writer: None,
        }
    }

    pub fn read(&mut self, buf: &mut [u8]) -> Result<uint, UvError> {
        // This read operation needs to get canceled on an unwind via libuv's
        // uv_read_stop function
        let _f = ForbidUnwind::new("stream read");

        let mut rcx = ReadContext {
            buf: Some(slice_to_uv_buf(buf)),
            // if the read is canceled, we'll see eof, otherwise this will get
            // overwritten
            result: 0,
            task: None,
        };
        // When reading a TTY stream on windows, libuv will invoke alloc_cb
        // immediately as part of the call to alloc_cb. What this means is that
        // we must be ready for this to happen (by setting the data in the uv
        // handle). In theory this otherwise doesn't need to happen until after
        // the read is successfully started.
        unsafe { uvll::set_data_for_uv_handle(self.handle, &mut rcx) }

        // Send off the read request, but don't block until we're sure that the
        // read request is queued.
        let ret = match unsafe {
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
        };
        // Make sure a read cancellation sees that there's no pending read
        unsafe { uvll::set_data_for_uv_handle(self.handle, 0 as *mut int) }
        return ret;
    }

    pub fn cancel_read(&mut self, reason: ssize_t) -> Option<BlockedTask> {
        // When we invoke uv_read_stop, it cancels the read and alloc
        // callbacks. We need to manually wake up a pending task (if one was
        // present).
        assert_eq!(unsafe { uvll::uv_read_stop(self.handle) }, 0);
        let data = unsafe {
            let data = uvll::get_data_for_uv_handle(self.handle);
            if data.is_null() { return None }
            uvll::set_data_for_uv_handle(self.handle, 0 as *mut int);
            &mut *(data as *mut ReadContext)
        };
        data.result = reason;
        data.task.take()
    }

    pub fn write(&mut self, buf: &[u8], may_timeout: bool) -> Result<(), UvError> {
        // The ownership of the write request is dubious if this function
        // unwinds. I believe that if the write_cb fails to re-schedule the task
        // then the write request will be leaked.
        let _f = ForbidUnwind::new("stream write");

        // Prepare the write request, either using a cached one or allocating a
        // new one
        let mut req = match self.last_write_req.take() {
            Some(req) => req, None => Request::new(uvll::UV_WRITE),
        };
        req.set_data(ptr::null_mut::<()>());

        // And here's where timeouts get a little interesting. Currently, libuv
        // does not support canceling an in-flight write request. Consequently,
        // when a write timeout expires, there's not much we can do other than
        // detach the sleeping task from the write request itself. Semantically,
        // this means that the write request will complete asynchronously, but
        // the calling task will return error (because the write timed out).
        //
        // There is special wording in the documentation of set_write_timeout()
        // indicating that this is a plausible failure scenario, and this
        // function is why that wording exists.
        //
        // Implementation-wise, we must be careful when passing a buffer down to
        // libuv. Most of this implementation avoids allocations because of the
        // blocking guarantee (all stack local variables are valid for the
        // entire read/write request). If our write request can be timed out,
        // however, we must heap allocate the data and pass that to the libuv
        // functions instead. The reason for this is that if we time out and
        // return, there's no guarantee that `buf` is a valid buffer any more.
        //
        // To do this, the write context has an optionally owned vector of
        // bytes.
        let data = if may_timeout {Some(Vec::from_slice(buf))} else {None};
        let uv_buf = if may_timeout {
            slice_to_uv_buf(data.as_ref().unwrap().as_slice())
        } else {
            slice_to_uv_buf(buf)
        };

        // Send off the request, but be careful to not block until we're sure
        // that the write request is queued. If the request couldn't be queued,
        // then we should return immediately with an error.
        match unsafe {
            uvll::uv_write(req.handle, self.handle, [uv_buf], write_cb)
        } {
            0 => {
                let mut wcx = WriteContext {
                    result: uvll::ECANCELED,
                    stream: self as *mut _,
                    data: data,
                };
                req.defuse(); // uv callback now owns this request

                let loop_ = unsafe { uvll::get_loop_for_uv_handle(self.handle) };
                wait_until_woken_after(&mut self.blocked_writer,
                                       &Loop::wrap(loop_), || {
                    req.set_data(&mut wcx);
                });

                if wcx.result != uvll::ECANCELED {
                    self.last_write_req = Some(Request::wrap(req.handle));
                    return match wcx.result {
                        0 => Ok(()),
                        n => Err(UvError(n)),
                    }
                }

                // This is the second case where canceling an in-flight write
                // gets interesting. If we've been canceled (no one reset our
                // result), then someone still needs to free the request, and
                // someone still needs to free the allocate buffer.
                //
                // To take care of this, we swap out the stack-allocated write
                // context for a heap-allocated context, transferring ownership
                // of everything to the write_cb. Libuv guarantees that this
                // callback will be invoked at some point, and the callback will
                // be responsible for deallocating these resources.
                //
                // Note that we don't cache this write request back in the
                // stream watcher because we no longer have ownership of it, and
                // we never will.
                let mut new_wcx = box WriteContext {
                    result: 0,
                    stream: 0 as *mut StreamWatcher,
                    data: wcx.data.take(),
                };
                unsafe {
                    req.set_data(&mut *new_wcx);
                    mem::forget(new_wcx);
                }
                Err(UvError(wcx.result))
            }
            n => Err(UvError(n)),
        }
    }

    pub fn cancel_write(&mut self) -> Option<BlockedTask> {
        self.blocked_writer.take()
    }
}

// This allocation callback expects to be invoked once and only once. It will
// unwrap the buffer in the ReadContext stored in the stream and return it. This
// will fail if it is called more than once.
extern fn alloc_cb(stream: *mut uvll::uv_stream_t, _hint: size_t, buf: *mut Buf) {
    uvdebug!("alloc_cb");
    unsafe {
        let rcx: &mut ReadContext =
            mem::transmute(uvll::get_data_for_uv_handle(stream));
        *buf = rcx.buf.take().expect("stream alloc_cb called more than once");
    }
}

// When a stream has read some data, we will always forcibly stop reading and
// return all the data read (even if it didn't fill the whole buffer).
extern fn read_cb(handle: *mut uvll::uv_stream_t, nread: ssize_t,
                  _buf: *const Buf) {
    uvdebug!("read_cb {}", nread);
    assert!(nread != uvll::ECANCELED as ssize_t);
    let rcx: &mut ReadContext = unsafe {
        mem::transmute(uvll::get_data_for_uv_handle(handle))
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
extern fn write_cb(req: *mut uvll::uv_write_t, status: c_int) {
    let mut req = Request::wrap(req);
    // Remember to not free the request because it is re-used between writes on
    // the same stream.
    let wcx: &mut WriteContext = unsafe { req.get_data() };
    wcx.result = status;

    // If the stream is present, we haven't timed out, otherwise we acquire
    // ownership of everything and then deallocate it all at once.
    if wcx.stream as uint != 0 {
        req.defuse();
        let stream: &mut StreamWatcher = unsafe { &mut *wcx.stream };
        wakeup(&mut stream.blocked_writer);
    } else {
        let _wcx: Box<WriteContext> = unsafe { mem::transmute(wcx) };
    }
}
