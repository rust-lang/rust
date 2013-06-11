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

Bindings to libuv, along with the default implementation of `core::rt::rtio`.

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

use container::Container;
use option::*;
use str::raw::from_c_str;
use to_str::ToStr;
use ptr::RawPtr;
use libc;
use vec;
use ptr;
use cast;
use str;
use libc::{c_void, c_int, size_t, malloc, free};
use cast::transmute;
use ptr::null;
use unstable::finally::Finally;

use rt::io::IoError;

#[cfg(test)] use unstable::run_in_bare_thread;

pub use self::file::FsRequest;
pub use self::net::{StreamWatcher, TcpWatcher};
pub use self::idle::IdleWatcher;
pub use self::timer::TimerWatcher;

/// The implementation of `rtio` for libuv
pub mod uvio;

/// C bindings to libuv
pub mod uvll;

pub mod file;
pub mod net;
pub mod idle;
pub mod timer;

/// XXX: Loop(*handle) is buggy with destructors. Normal structs
/// with dtors may not be destructured, but tuple structs can,
/// but the results are not correct.
pub struct Loop {
    handle: *uvll::uv_loop_t
}

/// The trait implemented by uv 'watchers' (handles). Watchers are
/// non-owning wrappers around the uv handles and are not completely
/// safe - there may be multiple instances for a single underlying
/// handle.  Watchers are generally created, then `start`ed, `stop`ed
/// and `close`ed, but due to their complex life cycle may not be
/// entirely memory safe if used in unanticipated patterns.
pub trait Watcher { }

pub trait Request { }

/// A type that wraps a native handle
pub trait NativeHandle<T> {
    pub fn from_native_handle(T) -> Self;
    pub fn native_handle(&self) -> T;
}

impl Loop {
    pub fn new() -> Loop {
        let handle = unsafe { uvll::loop_new() };
        assert!(handle.is_not_null());
        NativeHandle::from_native_handle(handle)
    }

    pub fn run(&mut self) {
        unsafe { uvll::run(self.native_handle()) };
    }

    pub fn close(&mut self) {
        unsafe { uvll::loop_delete(self.native_handle()) };
    }
}

impl NativeHandle<*uvll::uv_loop_t> for Loop {
    fn from_native_handle(handle: *uvll::uv_loop_t) -> Loop {
        Loop { handle: handle }
    }
    fn native_handle(&self) -> *uvll::uv_loop_t {
        self.handle
    }
}

// XXX: The uv alloc callback also has a *uv_handle_t arg
pub type AllocCallback = ~fn(uint) -> Buf;
pub type ReadCallback = ~fn(StreamWatcher, int, Buf, Option<UvError>);
pub type NullCallback = ~fn();
pub type IdleCallback = ~fn(IdleWatcher, Option<UvError>);
pub type ConnectionCallback = ~fn(StreamWatcher, Option<UvError>);
pub type FsCallback = ~fn(FsRequest, Option<UvError>);
pub type TimerCallback = ~fn(TimerWatcher, Option<UvError>);


/// Callbacks used by StreamWatchers, set as custom data on the foreign handle
struct WatcherData {
    read_cb: Option<ReadCallback>,
    write_cb: Option<ConnectionCallback>,
    connect_cb: Option<ConnectionCallback>,
    close_cb: Option<NullCallback>,
    alloc_cb: Option<AllocCallback>,
    idle_cb: Option<IdleCallback>,
    timer_cb: Option<TimerCallback>
}

pub trait WatcherInterop {
    fn event_loop(&self) -> Loop;
    fn install_watcher_data(&mut self);
    fn get_watcher_data<'r>(&'r mut self) -> &'r mut WatcherData;
    fn drop_watcher_data(&mut self);
}

impl<H, W: Watcher + NativeHandle<*H>> WatcherInterop for W {
    /// Get the uv event loop from a Watcher
    pub fn event_loop(&self) -> Loop {
        unsafe {
            let handle = self.native_handle();
            let loop_ = uvll::get_loop_for_uv_handle(handle);
            NativeHandle::from_native_handle(loop_)
        }
    }

    pub fn install_watcher_data(&mut self) {
        unsafe {
            let data = ~WatcherData {
                read_cb: None,
                write_cb: None,
                connect_cb: None,
                close_cb: None,
                alloc_cb: None,
                idle_cb: None,
                timer_cb: None
            };
            let data = transmute::<~WatcherData, *c_void>(data);
            uvll::set_data_for_uv_handle(self.native_handle(), data);
        }
    }

    pub fn get_watcher_data<'r>(&'r mut self) -> &'r mut WatcherData {
        unsafe {
            let data = uvll::get_data_for_uv_handle(self.native_handle());
            let data = transmute::<&*c_void, &mut ~WatcherData>(&data);
            return &mut **data;
        }
    }

    pub fn drop_watcher_data(&mut self) {
        unsafe {
            let data = uvll::get_data_for_uv_handle(self.native_handle());
            let _data = transmute::<*c_void, ~WatcherData>(data);
            uvll::set_data_for_uv_handle(self.native_handle(), null::<()>());
        }
    }
}

// XXX: Need to define the error constants like EOF so they can be
// compared to the UvError type

pub struct UvError(uvll::uv_err_t);

impl UvError {
    pub fn name(&self) -> ~str {
        unsafe {
            let inner = match self { &UvError(ref a) => a };
            let name_str = uvll::err_name(inner);
            assert!(name_str.is_not_null());
            from_c_str(name_str)
        }
    }

    pub fn desc(&self) -> ~str {
        unsafe {
            let inner = match self { &UvError(ref a) => a };
            let desc_str = uvll::strerror(inner);
            assert!(desc_str.is_not_null());
            from_c_str(desc_str)
        }
    }

    pub fn is_eof(&self) -> bool {
        self.code == uvll::EOF
    }
}

impl ToStr for UvError {
    fn to_str(&self) -> ~str {
        fmt!("%s: %s", self.name(), self.desc())
    }
}

#[test]
fn error_smoke_test() {
    let err = uvll::uv_err_t { code: 1, sys_errno_: 1 };
    let err: UvError = UvError(err);
    assert_eq!(err.to_str(), ~"EOF: end of file");
}

pub fn last_uv_error<H, W: Watcher + NativeHandle<*H>>(watcher: &W) -> UvError {
    unsafe {
        let loop_ = watcher.event_loop();
        UvError(uvll::last_error(loop_.native_handle()))
    }
}

pub fn uv_error_to_io_error(uverr: UvError) -> IoError {
    unsafe {
        // Importing error constants
        use rt::uv::uvll::*;
        use rt::io::*;

        // uv error descriptions are static
        let c_desc = uvll::strerror(&*uverr);
        let desc = str::raw::c_str_to_static_slice(c_desc);

        let kind = match uverr.code {
            UNKNOWN => OtherIoError,
            OK => OtherIoError,
            EOF => EndOfFile,
            EACCES => PermissionDenied,
            ECONNREFUSED => ConnectionRefused,
            ECONNRESET => ConnectionReset,
            EPIPE => BrokenPipe,
            _ => {
                rtdebug!("uverr.code %u", uverr.code as uint);
                // XXX: Need to map remaining uv error types
                OtherIoError
            }
        };

        IoError {
            kind: kind,
            desc: desc,
            detail: None
        }
    }
}

/// Given a uv handle, convert a callback status to a UvError
// XXX: Follow the pattern below by parameterizing over T: Watcher, not T
pub fn status_to_maybe_uv_error<T>(handle: *T, status: c_int) -> Option<UvError> {
    if status != -1 {
        None
    } else {
        unsafe {
            rtdebug!("handle: %x", handle as uint);
            let loop_ = uvll::get_loop_for_uv_handle(handle);
            rtdebug!("loop: %x", loop_ as uint);
            let err = uvll::last_error(loop_);
            Some(UvError(err))
        }
    }
}

/// The uv buffer type
pub type Buf = uvll::uv_buf_t;

/// Borrow a slice to a Buf
pub fn slice_to_uv_buf(v: &[u8]) -> Buf {
    let data = vec::raw::to_ptr(v);
    unsafe { uvll::buf_init(data, v.len()) }
}

// XXX: Do these conversions without copying

/// Transmute an owned vector to a Buf
pub fn vec_to_uv_buf(v: ~[u8]) -> Buf {
    unsafe {
        let data = malloc(v.len() as size_t) as *u8;
        assert!(data.is_not_null());
        do vec::as_imm_buf(v) |b, l| {
            let data = data as *mut u8;
            ptr::copy_memory(data, b, l)
        }
        uvll::buf_init(data, v.len())
    }
}

/// Transmute a Buf that was once a ~[u8] back to ~[u8]
pub fn vec_from_uv_buf(buf: Buf) -> Option<~[u8]> {
    if !(buf.len == 0 && buf.base.is_null()) {
        let v = unsafe { vec::from_buf(buf.base, buf.len as uint) };
        unsafe { free(buf.base as *c_void) };
        return Some(v);
    } else {
        // No buffer
        rtdebug!("No buffer!");
        return None;
    }
}

#[test]
fn test_slice_to_uv_buf() {
    let slice = [0, .. 20];
    let buf = slice_to_uv_buf(slice);

    assert!(buf.len == 20);

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

#[test]
#[ignore(reason = "valgrind - loop destroyed before watcher?")]
fn idle_new_then_close() {
    do run_in_bare_thread {
        let mut loop_ = Loop::new();
        let idle_watcher = { IdleWatcher::new(&mut loop_) };
        idle_watcher.close(||());
    }
}

#[test]
fn idle_smoke_test() {
    do run_in_bare_thread {
        let mut loop_ = Loop::new();
        let mut idle_watcher = { IdleWatcher::new(&mut loop_) };
        let mut count = 10;
        let count_ptr: *mut int = &mut count;
        do idle_watcher.start |idle_watcher, status| {
            let mut idle_watcher = idle_watcher;
            assert!(status.is_none());
            if unsafe { *count_ptr == 10 } {
                idle_watcher.stop();
                idle_watcher.close(||());
            } else {
                unsafe { *count_ptr = *count_ptr + 1; }
            }
        }
        loop_.run();
        loop_.close();
        assert_eq!(count, 10);
    }
}

#[test]
fn idle_start_stop_start() {
    do run_in_bare_thread {
        let mut loop_ = Loop::new();
        let mut idle_watcher = { IdleWatcher::new(&mut loop_) };
        do idle_watcher.start |idle_watcher, status| {
            let mut idle_watcher = idle_watcher;
            assert!(status.is_none());
            idle_watcher.stop();
            do idle_watcher.start |idle_watcher, status| {
                assert!(status.is_none());
                let mut idle_watcher = idle_watcher;
                idle_watcher.stop();
                idle_watcher.close(||());
            }
        }
        loop_.run();
        loop_.close();
    }
}
