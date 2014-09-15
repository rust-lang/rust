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

#![crate_name = "rustuv"]
#![experimental]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/master/",
       html_playground_url = "http://play.rust-lang.org/")]

#![feature(macro_rules, unsafe_destructor)]
#![deny(unused_result, unused_must_use)]
#![allow(visible_private_types)]

#![reexport_test_harness_main = "test_main"]

#[cfg(test)] extern crate green;
#[cfg(test)] extern crate debug;
#[cfg(test)] extern crate "rustuv" as realrustuv;
extern crate libc;
extern crate alloc;

use libc::{c_int, c_void};
use std::fmt;
use std::mem;
use std::ptr;
use std::string;
use std::rt::local::Local;
use std::rt::rtio;
use std::rt::rtio::{IoResult, IoError};
use std::rt::task::{BlockedTask, Task};
use std::task;

pub use self::async::AsyncWatcher;
pub use self::file::{FsRequest, FileWatcher};
pub use self::idle::IdleWatcher;
pub use self::net::{TcpWatcher, TcpListener, TcpAcceptor, UdpWatcher};
pub use self::pipe::{PipeWatcher, PipeListener, PipeAcceptor};
pub use self::process::Process;
pub use self::signal::SignalWatcher;
pub use self::timer::TimerWatcher;
pub use self::tty::TtyWatcher;

// Run tests with libgreen instead of libnative.
#[cfg(test)] #[start]
fn start(argc: int, argv: *const *const u8) -> int {
    green::start(argc, argv, event_loop, test_main)
}

mod macros;

mod access;
mod timeout;
mod homing;
mod queue;
mod rc;

pub mod uvio;
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

/// Creates a new event loop which is powered by libuv
///
/// This function is used in tandem with libgreen's `PoolConfig` type as a value
/// for the `event_loop_factory` field. Using this function as the event loop
/// factory will power programs with libuv and enable green threading.
///
/// # Example
///
/// ```
/// extern crate rustuv;
/// extern crate green;
///
/// #[start]
/// fn start(argc: int, argv: *const *const u8) -> int {
///     green::start(argc, argv, rustuv::event_loop, main)
/// }
///
/// fn main() {
///     // this code is running inside of a green task powered by libuv
/// }
/// ```
pub fn event_loop() -> Box<rtio::EventLoop + Send> {
    box uvio::UvEventLoop::new() as Box<rtio::EventLoop + Send>
}

/// A type that wraps a uv handle
pub trait UvHandle<T> {
    fn uv_handle(&self) -> *mut T;

    fn uv_loop(&self) -> Loop {
        Loop::wrap(unsafe { uvll::get_loop_for_uv_handle(self.uv_handle()) })
    }

    // FIXME(#8888) dummy self
    fn alloc(_: Option<Self>, ty: uvll::uv_handle_type) -> *mut T {
        unsafe {
            let handle = uvll::malloc_handle(ty);
            assert!(!handle.is_null());
            handle as *mut T
        }
    }

    unsafe fn from_uv_handle<'a>(h: &'a *mut T) -> &'a mut Self {
        mem::transmute(uvll::get_data_for_uv_handle(*h))
    }

    fn install(self: Box<Self>) -> Box<Self> {
        unsafe {
            let myptr = mem::transmute::<&Box<Self>, &*mut u8>(&self);
            uvll::set_data_for_uv_handle(self.uv_handle(), *myptr);
        }
        self
    }

    fn close_async_(&mut self) {
        // we used malloc to allocate all handles, so we must always have at
        // least a callback to free all the handles we allocated.
        extern fn close_cb(handle: *mut uvll::uv_handle_t) {
            unsafe { uvll::free_handle(handle) }
        }

        unsafe {
            uvll::set_data_for_uv_handle(self.uv_handle(), ptr::null_mut::<()>());
            uvll::uv_close(self.uv_handle() as *mut uvll::uv_handle_t, close_cb)
        }
    }

    fn close(&mut self) {
        let mut slot = None;

        unsafe {
            uvll::uv_close(self.uv_handle() as *mut uvll::uv_handle_t, close_cb);
            uvll::set_data_for_uv_handle(self.uv_handle(),
                                         ptr::null_mut::<()>());

            wait_until_woken_after(&mut slot, &self.uv_loop(), || {
                uvll::set_data_for_uv_handle(self.uv_handle(), &mut slot);
            })
        }

        extern fn close_cb(handle: *mut uvll::uv_handle_t) {
            unsafe {
                let data = uvll::get_data_for_uv_handle(handle);
                uvll::free_handle(handle);
                if data == ptr::null_mut() { return }
                let slot: &mut Option<BlockedTask> = mem::transmute(data);
                wakeup(slot);
            }
        }
    }
}

pub struct ForbidSwitch {
    msg: &'static str,
    io: uint,
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
                "didn't want a scheduler switch: {}",
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
                "didn't want an unwind during: {}", self.msg);
    }
}

fn wait_until_woken_after(slot: *mut Option<BlockedTask>,
                          loop_: &Loop,
                          f: ||) {
    let _f = ForbidUnwind::new("wait_until_woken_after");
    unsafe {
        assert!((*slot).is_none());
        let task: Box<Task> = Local::take();
        loop_.modify_blockers(1);
        task.deschedule(1, |task| {
            *slot = Some(task);
            f();
            Ok(())
        });
        loop_.modify_blockers(-1);
    }
}

fn wakeup(slot: &mut Option<BlockedTask>) {
    assert!(slot.is_some());
    let _ = slot.take().unwrap().wake().map(|t| t.reawaken());
}

pub struct Request {
    pub handle: *mut uvll::uv_req_t,
    defused: bool,
}

impl Request {
    pub fn new(ty: uvll::uv_req_type) -> Request {
        unsafe {
            let handle = uvll::malloc_req(ty);
            uvll::set_data_for_req(handle, ptr::null_mut::<()>());
            Request::wrap(handle)
        }
    }

    pub fn wrap(handle: *mut uvll::uv_req_t) -> Request {
        Request { handle: handle, defused: false }
    }

    pub fn set_data<T>(&self, t: *mut T) {
        unsafe { uvll::set_data_for_req(self.handle, t) }
    }

    pub unsafe fn get_data<T>(&self) -> &'static mut T {
        let data = uvll::get_data_for_req(self.handle);
        assert!(data != ptr::null_mut());
        mem::transmute(data)
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

/// FIXME: Loop(*handle) is buggy with destructors. Normal structs
/// with dtors may not be destructured, but tuple structs can,
/// but the results are not correct.
pub struct Loop {
    handle: *mut uvll::uv_loop_t
}

impl Loop {
    pub fn new() -> Loop {
        let handle = unsafe { uvll::loop_new() };
        assert!(handle.is_not_null());
        unsafe { uvll::set_data_for_uv_loop(handle, 0 as *mut c_void) }
        Loop::wrap(handle)
    }

    pub fn wrap(handle: *mut uvll::uv_loop_t) -> Loop { Loop { handle: handle } }

    pub fn run(&mut self) {
        assert_eq!(unsafe { uvll::uv_run(self.handle, uvll::RUN_DEFAULT) }, 0);
    }

    pub fn close(&mut self) {
        unsafe { uvll::uv_loop_delete(self.handle) };
    }

    // The 'data' field of the uv_loop_t is used to count the number of tasks
    // that are currently blocked waiting for I/O to complete.
    fn modify_blockers(&self, amt: uint) {
        unsafe {
            let cur = uvll::get_data_for_uv_loop(self.handle) as uint;
            uvll::set_data_for_uv_loop(self.handle, (cur + amt) as *mut c_void)
        }
    }

    fn get_blockers(&self) -> uint {
        unsafe { uvll::get_data_for_uv_loop(self.handle) as uint }
    }
}

// FIXME: Need to define the error constants like EOF so they can be
// compared to the UvError type

pub struct UvError(c_int);

impl UvError {
    pub fn name(&self) -> String {
        unsafe {
            let inner = match self { &UvError(a) => a };
            let name_str = uvll::uv_err_name(inner);
            assert!(name_str.is_not_null());
            string::raw::from_buf(name_str as *const u8)
        }
    }

    pub fn desc(&self) -> String {
        unsafe {
            let inner = match self { &UvError(a) => a };
            let desc_str = uvll::uv_strerror(inner);
            assert!(desc_str.is_not_null());
            string::raw::from_buf(desc_str as *const u8)
        }
    }

    pub fn is_eof(&self) -> bool {
        let UvError(handle) = *self;
        handle == uvll::EOF
    }
}

impl fmt::Show for UvError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.name(), self.desc())
    }
}

#[test]
fn error_smoke_test() {
    let err: UvError = UvError(uvll::EOF);
    assert_eq!(err.to_string(), "EOF: end of file".to_string());
}

#[cfg(unix)]
pub fn uv_error_to_io_error(uverr: UvError) -> IoError {
    let UvError(errcode) = uverr;
    IoError {
        code: if errcode == uvll::EOF {libc::EOF as uint} else {-errcode as uint},
        extra: 0,
        detail: Some(uverr.desc()),
    }
}

#[cfg(windows)]
pub fn uv_error_to_io_error(uverr: UvError) -> IoError {
    let UvError(errcode) = uverr;
    IoError {
        code: match errcode {
            uvll::EOF => libc::EOF,
            uvll::EACCES => libc::ERROR_ACCESS_DENIED,
            uvll::ECONNREFUSED => libc::WSAECONNREFUSED,
            uvll::ECONNRESET => libc::WSAECONNRESET,
            uvll::ENOTCONN => libc::WSAENOTCONN,
            uvll::ENOENT => libc::ERROR_FILE_NOT_FOUND,
            uvll::EPIPE => libc::ERROR_NO_DATA,
            uvll::ECONNABORTED => libc::WSAECONNABORTED,
            uvll::EADDRNOTAVAIL => libc::WSAEADDRNOTAVAIL,
            uvll::ECANCELED => libc::ERROR_OPERATION_ABORTED,
            uvll::EADDRINUSE => libc::WSAEADDRINUSE,
            uvll::EPERM => libc::ERROR_ACCESS_DENIED,
            err => {
                uvdebug!("uverr.code {}", err as int);
                // FIXME: Need to map remaining uv error types
                -1
            }
        } as uint,
        extra: 0,
        detail: Some(uverr.desc()),
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

pub fn status_to_io_result(status: c_int) -> IoResult<()> {
    if status >= 0 {Ok(())} else {Err(uv_error_to_io_error(UvError(status)))}
}

/// The uv buffer type
pub type Buf = uvll::uv_buf_t;

pub fn empty_buf() -> Buf {
    uvll::uv_buf_t {
        base: ptr::null_mut(),
        len: 0,
    }
}

/// Borrow a slice to a Buf
pub fn slice_to_uv_buf(v: &[u8]) -> Buf {
    let data = v.as_ptr();
    uvll::uv_buf_t { base: data as *mut u8, len: v.len() as uvll::uv_buf_len_t }
}

// This function is full of lies!
#[cfg(test)]
fn local_loop() -> &'static mut uvio::UvIoFactory {
    use std::raw::TraitObject;
    unsafe {
        mem::transmute({
            let mut task = Local::borrow(None::<Task>);
            let mut io = task.local_io().unwrap();
            let obj: TraitObject =
                mem::transmute(io.get());
            obj.data
        })
    }
}

#[cfg(test)]
fn next_test_ip4() -> std::rt::rtio::SocketAddr {
    use std::io;
    use std::rt::rtio;

    let io::net::ip::SocketAddr { ip, port } = io::test::next_test_ip4();
    let ip = match ip {
        io::net::ip::Ipv4Addr(a, b, c, d) => rtio::Ipv4Addr(a, b, c, d),
        _ => unreachable!(),
    };
    rtio::SocketAddr { ip: ip, port: port }
}

#[cfg(test)]
fn next_test_ip6() -> std::rt::rtio::SocketAddr {
    use std::io;
    use std::rt::rtio;

    let io::net::ip::SocketAddr { ip, port } = io::test::next_test_ip6();
    let ip = match ip {
        io::net::ip::Ipv6Addr(a, b, c, d, e, f, g, h) =>
            rtio::Ipv6Addr(a, b, c, d, e, f, g, h),
        _ => unreachable!(),
    };
    rtio::SocketAddr { ip: ip, port: port }
}

#[cfg(test)]
mod test {
    use std::mem::transmute;
    use std::rt::thread::Thread;

    use super::{slice_to_uv_buf, Loop};

    #[test]
    fn test_slice_to_uv_buf() {
        let slice = [0, .. 20];
        let buf = slice_to_uv_buf(slice);

        assert_eq!(buf.len, 20);

        unsafe {
            let base = transmute::<*mut u8, *mut u8>(buf.base);
            (*base) = 1;
            (*base.offset(1)) = 2;
        }

        assert!(slice[0] == 1);
        assert!(slice[1] == 2);
    }


    #[test]
    fn loop_smoke_test() {
        Thread::start(proc() {
            let mut loop_ = Loop::new();
            loop_.run();
            loop_.close();
        }).join();
    }
}
