// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Bindings to libuv, along with the default implementation of `std::rt::rtio`.

UV types consist of the event loop (Loop), Watchers, Requests and
Callbacks.

Watchers and Requests encapsulate pointers to uv *handles*, which have
subtyping relationships with each other.  This subtyping is reflected
in the bindings with explicit or implicit coercions. For example, an
upcast from TcpWatcher to StreamWatcher is done with
`tcp_watcher.as_stream()`. In other cases a callback on a specific
type of watcher will be passed a watcher of a supertype.

Currently all use of Request types (connect/write requests) are
encapsulated in the bindings and don't need to be dealt with by the
caller.

# Safety note

Due to the complex lifecycle of uv handles, as well as compiler bugs,
this module is not memory safe and requires explicit memory management,
via `close` and `delete` methods.

*/

#[crate_id = "rustuv#0.9"];
#[license = "MIT/ASL2"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];

#[feature(macro_rules)];

#[cfg(test)] extern mod green;

use std::cast;
use std::io;
use std::io::IoError;
use std::libc::{c_int, malloc};
use std::ptr::null;
use std::ptr;
use std::rt::local::Local;
use std::rt::task::{BlockedTask, Task};
use std::str::raw::from_c_str;
use std::str;
use std::task;
use std::unstable::finally::Finally;

pub use self::async::AsyncWatcher;
pub use self::file::{FsRequest, FileWatcher};
pub use self::idle::IdleWatcher;
pub use self::net::{TcpWatcher, TcpListener, TcpAcceptor, UdpWatcher};
pub use self::pipe::{PipeWatcher, PipeListener, PipeAcceptor};
pub use self::process::Process;
pub use self::signal::SignalWatcher;
pub use self::timer::TimerWatcher;
pub use self::tty::TtyWatcher;

mod macros;

mod queue;
mod homing;

/// The implementation of `rtio` for libuv
pub mod uvio;

/// C bindings to libuv
pub mod uvll;

pub mod file;
pub mod net;
pub mod idle;
pub mod timer;
pub mod async;
pub mod addrinfo;
pub mod process;
pub mod pipe;
pub mod tty;
pub mod signal;
pub mod stream;

/// A type that wraps a uv handle
pub trait UvHandle<T> {
    fn uv_handle(&self) -> *T;

    // FIXME(#8888) dummy self
    fn alloc(_: Option<Self>, ty: uvll::uv_handle_type) -> *T {
        unsafe {
            let handle = uvll::malloc_handle(ty);
            assert!(!handle.is_null());
            handle as *T
        }
    }

    unsafe fn from_uv_handle<'a>(h: &'a *T) -> &'a mut Self {
        cast::transmute(uvll::get_data_for_uv_handle(*h))
    }

    fn install(~self) -> ~Self {
        unsafe {
            let myptr = cast::transmute::<&~Self, &*u8>(&self);
            uvll::set_data_for_uv_handle(self.uv_handle(), *myptr);
        }
        self
    }

    fn close_async_(&mut self) {
        // we used malloc to allocate all handles, so we must always have at
        // least a callback to free all the handles we allocated.
        extern fn close_cb(handle: *uvll::uv_handle_t) {
            unsafe { uvll::free_handle(handle) }
        }

        unsafe {
            uvll::set_data_for_uv_handle(self.uv_handle(), null::<()>());
            uvll::uv_close(self.uv_handle() as *uvll::uv_handle_t, close_cb)
        }
    }

    fn close(&mut self) {
        let mut slot = None;

        unsafe {
            uvll::uv_close(self.uv_handle() as *uvll::uv_handle_t, close_cb);
            uvll::set_data_for_uv_handle(self.uv_handle(), ptr::null::<()>());

            wait_until_woken_after(&mut slot, || {
                uvll::set_data_for_uv_handle(self.uv_handle(), &slot);
            })
        }

        extern fn close_cb(handle: *uvll::uv_handle_t) {
            unsafe {
                let data = uvll::get_data_for_uv_handle(handle);
                uvll::free_handle(handle);
                if data == ptr::null() { return }
                let slot: &mut Option<BlockedTask> = cast::transmute(data);
                wakeup(slot);
            }
        }
    }
}

pub struct ForbidSwitch {
    priv msg: &'static str,
    priv io: uint,
}

impl ForbidSwitch {
    fn new(s: &'static str) -> ForbidSwitch {
        ForbidSwitch {
            msg: s,
            io: homing::local_id(),
        }
    }
}

impl Drop for ForbidSwitch {
    fn drop(&mut self) {
        assert!(self.io == homing::local_id(),
                "didnt want a scheduler switch: {}",
                self.msg);
    }
}

pub struct ForbidUnwind {
    msg: &'static str,
    failing_before: bool,
}

impl ForbidUnwind {
    fn new(s: &'static str) -> ForbidUnwind {
        ForbidUnwind {
            msg: s, failing_before: task::failing(),
        }
    }
}

impl Drop for ForbidUnwind {
    fn drop(&mut self) {
        assert!(self.failing_before == task::failing(),
                "didnt want an unwind during: {}", self.msg);
    }
}

fn wait_until_woken_after(slot: *mut Option<BlockedTask>, f: ||) {
    let _f = ForbidUnwind::new("wait_until_woken_after");
    unsafe {
        assert!((*slot).is_none());
        let task: ~Task = Local::take();
        task.deschedule(1, |task| {
            *slot = Some(task);
            f();
            Ok(())
        });
    }
}

fn wakeup(slot: &mut Option<BlockedTask>) {
    assert!(slot.is_some());
    slot.take_unwrap().wake().map(|t| t.reawaken(true));
}

pub struct Request {
    handle: *uvll::uv_req_t,
    priv defused: bool,
}

impl Request {
    pub fn new(ty: uvll::uv_req_type) -> Request {
        unsafe {
            let handle = uvll::malloc_req(ty);
            uvll::set_data_for_req(handle, null::<()>());
            Request::wrap(handle)
        }
    }

    pub fn wrap(handle: *uvll::uv_req_t) -> Request {
        Request { handle: handle, defused: false }
    }

    pub fn set_data<T>(&self, t: *T) {
        unsafe { uvll::set_data_for_req(self.handle, t) }
    }

    pub unsafe fn get_data<T>(&self) -> &'static mut T {
        let data = uvll::get_data_for_req(self.handle);
        assert!(data != null());
        cast::transmute(data)
    }

    // This function should be used when the request handle has been given to an
    // underlying uv function, and the uv function has succeeded. This means
    // that uv will at some point invoke the callback, and in the meantime we
    // can't deallocate the handle because libuv could be using it.
    //
    // This is still a problem in blocking situations due to linked failure. In
    // the connection callback the handle should be re-wrapped with the `wrap`
    // function to ensure its destruction.
    pub fn defuse(&mut self) {
        self.defused = true;
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        if !self.defused {
            unsafe { uvll::free_req(self.handle) }
        }
    }
}

/// XXX: Loop(*handle) is buggy with destructors. Normal structs
/// with dtors may not be destructured, but tuple structs can,
/// but the results are not correct.
pub struct Loop {
    priv handle: *uvll::uv_loop_t
}

impl Loop {
    pub fn new() -> Loop {
        let handle = unsafe { uvll::loop_new() };
        assert!(handle.is_not_null());
        Loop::wrap(handle)
    }

    pub fn wrap(handle: *uvll::uv_loop_t) -> Loop { Loop { handle: handle } }

    pub fn run(&mut self) {
        unsafe { uvll::uv_run(self.handle, uvll::RUN_DEFAULT) };
    }

    pub fn close(&mut self) {
        unsafe { uvll::uv_loop_delete(self.handle) };
    }
}

// XXX: Need to define the error constants like EOF so they can be
// compared to the UvError type

pub struct UvError(c_int);

impl UvError {
    pub fn name(&self) -> ~str {
        unsafe {
            let inner = match self { &UvError(a) => a };
            let name_str = uvll::uv_err_name(inner);
            assert!(name_str.is_not_null());
            from_c_str(name_str)
        }
    }

    pub fn desc(&self) -> ~str {
        unsafe {
            let inner = match self { &UvError(a) => a };
            let desc_str = uvll::uv_strerror(inner);
            assert!(desc_str.is_not_null());
            from_c_str(desc_str)
        }
    }

    pub fn is_eof(&self) -> bool {
        **self == uvll::EOF
    }
}

impl ToStr for UvError {
    fn to_str(&self) -> ~str {
        format!("{}: {}", self.name(), self.desc())
    }
}

#[test]
fn error_smoke_test() {
    let err: UvError = UvError(uvll::EOF);
    assert_eq!(err.to_str(), ~"EOF: end of file");
}

pub fn uv_error_to_io_error(uverr: UvError) -> IoError {
    unsafe {
        // Importing error constants

        // uv error descriptions are static
        let c_desc = uvll::uv_strerror(*uverr);
        let desc = str::raw::c_str_to_static_slice(c_desc);

        let kind = match *uverr {
            uvll::UNKNOWN => io::OtherIoError,
            uvll::OK => io::OtherIoError,
            uvll::EOF => io::EndOfFile,
            uvll::EACCES => io::PermissionDenied,
            uvll::ECONNREFUSED => io::ConnectionRefused,
            uvll::ECONNRESET => io::ConnectionReset,
            uvll::ENOTCONN => io::NotConnected,
            uvll::ENOENT => io::FileNotFound,
            uvll::EPIPE => io::BrokenPipe,
            uvll::ECONNABORTED => io::ConnectionAborted,
            uvll::EADDRNOTAVAIL => io::ConnectionRefused,
            err => {
                uvdebug!("uverr.code {}", err as int);
                // XXX: Need to map remaining uv error types
                io::OtherIoError
            }
        };

        IoError {
            kind: kind,
            desc: desc,
            detail: None
        }
    }
}

/// Given a uv error code, convert a callback status to a UvError
pub fn status_to_maybe_uv_error(status: c_int) -> Option<UvError> {
    if status >= 0 {
        None
    } else {
        Some(UvError(status))
    }
}

pub fn status_to_io_result(status: c_int) -> Result<(), IoError> {
    if status >= 0 {Ok(())} else {Err(uv_error_to_io_error(UvError(status)))}
}

/// The uv buffer type
pub type Buf = uvll::uv_buf_t;

pub fn empty_buf() -> Buf {
    uvll::uv_buf_t {
        base: null(),
        len: 0,
    }
}

/// Borrow a slice to a Buf
pub fn slice_to_uv_buf(v: &[u8]) -> Buf {
    let data = v.as_ptr();
    uvll::uv_buf_t { base: data, len: v.len() as uvll::uv_buf_len_t }
}

// This function is full of lies!
#[cfg(test)]
fn local_loop() -> &'static mut uvio::UvIoFactory {
    unsafe {
        cast::transmute({
            let mut task = Local::borrow(None::<Task>);
            let mut io = task.get().local_io().unwrap();
            let (_vtable, uvio): (uint, &'static mut uvio::UvIoFactory) =
                cast::transmute(io.get());
            uvio
        })
    }
}

#[cfg(test)]
mod test {
    use std::cast::transmute;
    use std::ptr;
    use std::unstable::run_in_bare_thread;

    use super::{slice_to_uv_buf, Loop};

    #[test]
    fn test_slice_to_uv_buf() {
        let slice = [0, .. 20];
        let buf = slice_to_uv_buf(slice);

        assert_eq!(buf.len, 20);

        unsafe {
            let base = transmute::<*u8, *mut u8>(buf.base);
            (*base) = 1;
            (*ptr::mut_offset(base, 1)) = 2;
        }

        assert!(slice[0] == 1);
        assert!(slice[1] == 2);
    }


    #[test]
    fn loop_smoke_test() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            loop_.run();
            loop_.close();
        }
    }
}
