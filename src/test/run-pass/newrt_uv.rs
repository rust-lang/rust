// xfail-test not a test

/*!

Bindings to libuv.

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

use core::private::run_in_bare_thread;
use core::str::raw::from_c_str;
use core::libc::{c_void, c_int, size_t, malloc, free, ssize_t};
use core::cast::transmute;
use core::ptr::null;
use core::sys::size_of;
use std::net::ip::*;
use std::cell::Cell;
use uvll = std::uv::ll;
use super::thread::Thread;

// A little helper. Can't figure out how to do this with net::ip
pub fn ip4addr(addr: &str, port: uint) -> IpAddr {
    Ipv4(unsafe { uvll::ip4_addr(addr, port as int) })
}

/// A trait for callbacks to implement. Provides a little extra type safety
/// for generic, unsafe interop functions like `set_watcher_callback`.
trait Callback { }

type NullCallback = ~fn();
impl NullCallback: Callback { }

/// A type that wraps a native handle
trait NativeHandle<T> {
    static pub fn from_native_handle(T) -> Self;
    pub fn native_handle(&self) -> T;
}

/// FIXME: Loop(*handle) is buggy with destructors. Normal structs
/// with dtors may not be destructured, but tuple structs can,
/// but the results are not correct.
pub struct Loop {
    handle: *uvll::uv_loop_t
}

pub impl Loop {
    static fn new() -> Loop {
        let handle = unsafe { uvll::loop_new() };
        assert handle.is_not_null();
        NativeHandle::from_native_handle(handle)
    }

    fn run(&mut self) {
        unsafe { uvll::run(self.native_handle()) };
    }

    fn close(&mut self) {
        unsafe { uvll::loop_delete(self.native_handle()) };
    }
}

pub impl Loop: NativeHandle<*uvll::uv_loop_t> {
    static fn from_native_handle(handle: *uvll::uv_loop_t) -> Loop {
        Loop { handle: handle }
    }
    fn native_handle(&self) -> *uvll::uv_loop_t {
        self.handle
    }
}

/// The trait implemented by uv 'watchers' (handles). Watchers
/// are non-owning wrappers around the uv handles and are not completely safe -
/// there may be multiple instances for a single underlying handle.
/// Watchers are generally created, then `start`ed, `stop`ed and `close`ed,
/// but due to their complex life cycle may not be entirely memory safe
/// if used in unanticipated patterns.
trait Watcher {
    fn event_loop(&self) -> Loop;
}

pub struct IdleWatcher(*uvll::uv_idle_t);

impl IdleWatcher: Watcher {
    fn event_loop(&self) -> Loop {
        loop_from_watcher(self)
    }
}

type IdleCallback = ~fn(IdleWatcher, Option<UvError>);
impl IdleCallback: Callback { }

pub impl IdleWatcher {
    static fn new(loop_: &mut Loop) -> IdleWatcher {
        unsafe {
            let handle = uvll::idle_new();
            assert handle.is_not_null();
            assert 0 == uvll::idle_init(loop_.native_handle(), handle);
            uvll::set_data_for_uv_handle(handle, null::<()>());
            NativeHandle::from_native_handle(handle)
        }
    }

    fn start(&mut self, cb: IdleCallback) {

        set_watcher_callback(self, cb);
        unsafe { assert 0 == uvll::idle_start(self.native_handle(), idle_cb) };

        extern fn idle_cb(handle: *uvll::uv_idle_t, status: c_int) {
            let idle_watcher: IdleWatcher = NativeHandle::from_native_handle(handle);
            let cb: &IdleCallback = borrow_callback_from_watcher(&idle_watcher);
            let status = status_to_maybe_uv_error(handle, status);
            (*cb)(idle_watcher, status);
        }
    }

    fn stop(&mut self) {
        unsafe { assert 0 == uvll::idle_stop(self.native_handle()); }
    }

    fn close(self) {
        unsafe { uvll::close(self.native_handle(), close_cb) };

        extern fn close_cb(handle: *uvll::uv_idle_t) {
            let mut idle_watcher = NativeHandle::from_native_handle(handle);
            drop_watcher_callback::<uvll::uv_idle_t, IdleWatcher, IdleCallback>(&mut idle_watcher);
            unsafe { uvll::idle_delete(handle) };
        }
    }
}

pub impl IdleWatcher: NativeHandle<*uvll::uv_idle_t> {
    static fn from_native_handle(handle: *uvll::uv_idle_t) -> IdleWatcher {
        IdleWatcher(handle)
    }
    fn native_handle(&self) -> *uvll::uv_idle_t {
        match self { &IdleWatcher(ptr) => ptr }
    }
}

// uv_stream t is the parent class of uv_tcp_t, uv_pipe_t, uv_tty_t and uv_file_t
pub struct StreamWatcher(*uvll::uv_stream_t);

impl StreamWatcher: Watcher {
    fn event_loop(&self) -> Loop {
        loop_from_watcher(self)
    }
}

type ReadCallback = ~fn(StreamWatcher, int, Buf, Option<UvError>);
impl ReadCallback: Callback { }

// XXX: The uv alloc callback also has a *uv_handle_t arg
type AllocCallback = ~fn(uint) -> Buf;
impl AllocCallback: Callback { }

pub impl StreamWatcher {

    fn read_start(&mut self, alloc: AllocCallback, cb: ReadCallback) {
        let data = get_watcher_data(self);
        data.alloc_cb = Some(alloc);
        data.read_cb = Some(cb);

        let handle = self.native_handle();
        unsafe { uvll::read_start(handle, alloc_cb, read_cb); }

        extern fn alloc_cb(stream: *uvll::uv_stream_t, suggested_size: size_t) -> Buf {
            let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(stream);
            let data = get_watcher_data(&mut stream_watcher);
            let alloc_cb = data.alloc_cb.get_ref();
            return (*alloc_cb)(suggested_size as uint);
        }

        extern fn read_cb(stream: *uvll::uv_stream_t, nread: ssize_t, ++buf: Buf) {
            rtdebug!("read_cb");
            rtdebug!("buf addr: %x", buf.base as uint);
            rtdebug!("buf len: %d", buf.len as int);
            let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(stream);
            let data = get_watcher_data(&mut stream_watcher);
            let cb = data.read_cb.get_ref();
            let status = status_to_maybe_uv_error(stream, nread as c_int);
            (*cb)(stream_watcher, nread as int, buf, status);
        }
    }

    fn read_stop(&mut self) {
        // It would be nice to drop the alloc and read callbacks here, but read_stop
        // may be called from inside one of them and we would end up freeing the
        // in-use environment
        let handle = self.native_handle();
        unsafe { uvll::read_stop(handle); }
    }

    // FIXME: Needs to take &[u8], not ~[u8]
    fn write(&mut self, msg: ~[u8], cb: ConnectionCallback) {
        let data = get_watcher_data(self);
        assert data.write_cb.is_none();
        data.write_cb = Some(cb);

        let req = WriteRequest::new();
        let buf = vec_to_uv_buf(msg);
        // FIXME: Allocation
        let bufs = ~[buf];
        unsafe { assert 0 == uvll::write(req.native_handle(),
                                         self.native_handle(),
                                         &bufs, write_cb); }
        // XXX: Freeing immediately after write. Is this ok?
        let _v = vec_from_uv_buf(buf);

        extern fn write_cb(req: *uvll::uv_write_t, status: c_int) {
            let write_request: WriteRequest = NativeHandle::from_native_handle(req);
            let mut stream_watcher = write_request.stream();
            write_request.delete();
            let cb = get_watcher_data(&mut stream_watcher).write_cb.swap_unwrap();
            let status = status_to_maybe_uv_error(stream_watcher.native_handle(), status);
            cb(stream_watcher, status);
        }
    }

    fn accept(&mut self, stream: StreamWatcher) {
        let self_handle = self.native_handle() as *c_void;
        let stream_handle = stream.native_handle() as *c_void;
        unsafe { assert 0 == uvll::accept(self_handle, stream_handle); }
    }

    fn close(self, cb: NullCallback) {
        {
            let mut self = self;
            let data = get_watcher_data(&mut self);
            assert data.close_cb.is_none();
            data.close_cb = Some(cb);
        }

        unsafe { uvll::close(self.native_handle(), close_cb); }

        extern fn close_cb(handle: *uvll::uv_stream_t) {
            let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(handle);
            {
                let mut data = get_watcher_data(&mut stream_watcher);
                data.close_cb.swap_unwrap()();
            }
            drop_watcher_data(&mut stream_watcher);
            unsafe { free(handle as *c_void) }
        }
    }
}

pub impl StreamWatcher: NativeHandle<*uvll::uv_stream_t> {
    static fn from_native_handle(handle: *uvll::uv_stream_t) -> StreamWatcher {
        StreamWatcher(handle)
    }
    fn native_handle(&self) -> *uvll::uv_stream_t {
        match self { &StreamWatcher(ptr) => ptr }
    }
}

pub struct TcpWatcher(*uvll::uv_tcp_t);

impl TcpWatcher: Watcher {
    fn event_loop(&self) -> Loop {
        loop_from_watcher(self)
    }
}

type ConnectionCallback = ~fn(StreamWatcher, Option<UvError>);
impl ConnectionCallback: Callback { }

pub impl TcpWatcher {
    static fn new(loop_: &mut Loop) -> TcpWatcher {
        unsafe {
            let handle = malloc(size_of::<uvll::uv_tcp_t>() as size_t) as *uvll::uv_tcp_t;
            assert handle.is_not_null();
            assert 0 == uvll::tcp_init(loop_.native_handle(), handle);
            let mut watcher = NativeHandle::from_native_handle(handle);
            install_watcher_data(&mut watcher);
            return watcher;
        }
    }

    fn bind(&mut self, address: IpAddr) {
        match address {
            Ipv4(addr) => {
                let result = unsafe { uvll::tcp_bind(self.native_handle(), &addr) };
                // FIXME: bind is likely to fail. need real error handling
                assert result == 0;
            }
            _ => fail!()
        }
    }

    fn connect(&mut self, address: IpAddr, cb: ConnectionCallback) {
        unsafe {
            assert get_watcher_data(self).connect_cb.is_none();
            get_watcher_data(self).connect_cb = Some(cb);

            let mut connect_watcher = ConnectRequest::new();
            let connect_handle = connect_watcher.native_handle();
            match address {
                Ipv4(addr) => {
                    rtdebug!("connect_t: %x", connect_handle as uint);
                    assert 0 == uvll::tcp_connect(connect_handle, self.native_handle(), &addr, connect_cb);
                }
                _ => fail!()
            }

            extern fn connect_cb(req: *uvll::uv_connect_t, status: c_int) {
                rtdebug!("connect_t: %x", req as uint);
                let connect_request: ConnectRequest = NativeHandle::from_native_handle(req);
                let mut stream_watcher = connect_request.stream();
                connect_request.delete();
                let cb: ConnectionCallback = get_watcher_data(&mut stream_watcher).connect_cb.swap_unwrap();
                let status = status_to_maybe_uv_error(stream_watcher.native_handle(), status);
                cb(stream_watcher, status);
            }
        }
    }

    fn listen(&mut self, cb: ConnectionCallback) {
        let data = get_watcher_data(self);
        assert data.connect_cb.is_none();
        data.connect_cb = Some(cb);

        unsafe {
            const BACKLOG: c_int = 128; // FIXME
            // FIXME: This can probably fail
            assert 0 == uvll::listen(self.native_handle(), BACKLOG, connection_cb);
        }

        extern fn connection_cb(handle: *uvll::uv_stream_t, status: c_int) {
            rtdebug!("connection_cb");
            let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(handle);
            let cb = get_watcher_data(&mut stream_watcher).connect_cb.swap_unwrap();
            let status = status_to_maybe_uv_error(stream_watcher.native_handle(), status);
            cb(stream_watcher, status);
        }
    }

    fn as_stream(&self) -> StreamWatcher {
        NativeHandle::from_native_handle(self.native_handle() as *uvll::uv_stream_t)
    }
}

pub impl TcpWatcher: NativeHandle<*uvll::uv_tcp_t> {
    static fn from_native_handle(handle: *uvll::uv_tcp_t) -> TcpWatcher {
        TcpWatcher(handle)
    }
    fn native_handle(&self) -> *uvll::uv_tcp_t {
        match self { &TcpWatcher(ptr) => ptr }
    }
}

trait Request { }

type ConnectCallback = ~fn(ConnectRequest, Option<UvError>);
impl ConnectCallback: Callback { }

// uv_connect_t is a subclass of uv_req_t
struct ConnectRequest(*uvll::uv_connect_t);

impl ConnectRequest: Request { }

impl ConnectRequest {

    static fn new() -> ConnectRequest {
        let connect_handle = unsafe { malloc(size_of::<uvll::uv_connect_t>() as size_t) };
        assert connect_handle.is_not_null();
        let connect_handle = connect_handle as *uvll::uv_connect_t;
        ConnectRequest(connect_handle)
    }

    fn stream(&self) -> StreamWatcher {
        unsafe {
            let stream_handle = uvll::get_stream_handle_from_connect_req(self.native_handle());
            NativeHandle::from_native_handle(stream_handle)
        }
    }

    fn delete(self) {
        unsafe { free(self.native_handle() as *c_void) }
    }
}

pub impl ConnectRequest: NativeHandle<*uvll::uv_connect_t> {
    static fn from_native_handle(handle: *uvll:: uv_connect_t) -> ConnectRequest {
        ConnectRequest(handle)
    }
    fn native_handle(&self) -> *uvll::uv_connect_t {
        match self { &ConnectRequest(ptr) => ptr }
    }
}

pub struct WriteRequest(*uvll::uv_write_t);

impl WriteRequest: Request { }

impl WriteRequest {

    static fn new() -> WriteRequest {
        let write_handle = unsafe { malloc(size_of::<uvll::uv_write_t>() as size_t) };
        assert write_handle.is_not_null();
        let write_handle = write_handle as *uvll::uv_write_t;
        WriteRequest(write_handle)
    }

    fn stream(&self) -> StreamWatcher {
        unsafe {
            let stream_handle = uvll::get_stream_handle_from_write_req(self.native_handle());
            NativeHandle::from_native_handle(stream_handle)
        }
    }

    fn delete(self) {
        unsafe { free(self.native_handle() as *c_void) }
    }
}

pub impl WriteRequest: NativeHandle<*uvll::uv_write_t> {
    static fn from_native_handle(handle: *uvll:: uv_write_t) -> WriteRequest {
        WriteRequest(handle)
    }
    fn native_handle(&self) -> *uvll::uv_write_t {
        match self { &WriteRequest(ptr) => ptr }
    }
}

// FIXME: Need to define the error constants like EOF so they can be compared
// to the UvError type

struct UvError(uvll::uv_err_t);

impl UvError {

    pure fn name(&self) -> ~str {
        unsafe { 
            let inner = match self { &UvError(ref a) => a };
            let name_str = uvll::err_name(inner);
            assert name_str.is_not_null();
            from_c_str(name_str)
        }
    }

    pure fn desc(&self) -> ~str {
        unsafe {
            let inner = match self { &UvError(ref a) => a };
            let desc_str = uvll::strerror(inner);
            assert desc_str.is_not_null();
            from_c_str(desc_str)
        }
    }
}

impl UvError: ToStr {
    pure fn to_str(&self) -> ~str {
        fmt!("%s: %s", self.name(), self.desc())
    }
}

#[test]
fn error_smoke_test() {
    let err = uvll::uv_err_t { code: 1, sys_errno_: 1 };
    let err: UvError = UvError(err);
    assert err.to_str() == ~"EOF: end of file";
}


/// Given a uv handle, convert a callback status to a UvError
// FIXME: Follow the pattern below by parameterizing over T: Watcher, not T
fn status_to_maybe_uv_error<T>(handle: *T, status: c_int) -> Option<UvError> {
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

/// Get the uv event loop from a Watcher
fn loop_from_watcher<H, W: Watcher NativeHandle<*H>>(watcher: &W) -> Loop {
    let handle = watcher.native_handle();
    let loop_ = unsafe { uvll::get_loop_for_uv_handle(handle) };
    NativeHandle::from_native_handle(loop_)
}

/// Set the custom data on a handle to a callback
/// Note: This is only suitable for watchers that make just one type of callback.
/// For others use WatcherData
fn set_watcher_callback<H, W: Watcher NativeHandle<*H>, CB: Callback>(watcher: &mut W, cb: CB) {
    drop_watcher_callback::<H, W, CB>(watcher);
    // NOTE: Boxing the callback so it fits into a pointer. Unfortunate extra allocation
    let boxed_cb = ~cb;
    let data = unsafe { transmute::<~CB, *c_void>(boxed_cb) };
    unsafe { uvll::set_data_for_uv_handle(watcher.native_handle(), data) };
}

/// Delete a callback from a handle's custom data
fn drop_watcher_callback<H, W: Watcher NativeHandle<*H>, CB: Callback>(watcher: &mut W) {
    unsafe {
        let handle = watcher.native_handle();
        let handle_data: *c_void = uvll::get_data_for_uv_handle(handle);
        if handle_data.is_not_null() {
            // Take ownership of the callback and drop it
            let _cb = transmute::<*c_void, ~CB>(handle_data);
            // Make sure the pointer is zeroed
            uvll::set_data_for_uv_handle(watcher.native_handle(), null::<()>());
        }
    }
}

/// Take a pointer to the callback installed as custom data
fn borrow_callback_from_watcher<H, W: Watcher NativeHandle<*H>, CB: Callback>(watcher: &W) -> &CB {
    unsafe {
        let handle = watcher.native_handle();
        let handle_data: *c_void = uvll::get_data_for_uv_handle(handle);
        assert handle_data.is_not_null();
        let cb = transmute::<&*c_void, &~CB>(&handle_data);
        return &**cb;
    }
}

/// Take ownership of the callback installed as custom data
fn take_callback_from_watcher<H, W: Watcher NativeHandle<*H>, CB: Callback>(watcher: &mut W) -> CB {
    unsafe {
        let handle = watcher.native_handle();
        let handle_data: *c_void = uvll::get_data_for_uv_handle(handle);
        assert handle_data.is_not_null();
        uvll::set_data_for_uv_handle(handle, null::<()>());
        let cb: ~CB = transmute::<*c_void, ~CB>(handle_data);
        let cb = match cb { ~cb => cb };
        return cb;
    }
}

/// Callbacks used by StreamWatchers, set as custom data on the foreign handle
struct WatcherData {
    read_cb: Option<ReadCallback>,
    write_cb: Option<ConnectionCallback>,
    connect_cb: Option<ConnectionCallback>,
    close_cb: Option<NullCallback>,
    alloc_cb: Option<AllocCallback>
}

fn install_watcher_data<H, W: Watcher NativeHandle<*H>>(watcher: &mut W) {
    unsafe {
        let data = ~WatcherData {
            read_cb: None,
            write_cb: None,
            connect_cb: None,
            close_cb: None,
            alloc_cb: None
        };
        let data = transmute::<~WatcherData, *c_void>(data);
        uvll::set_data_for_uv_handle(watcher.native_handle(), data);
    }
}

fn get_watcher_data<H, W: Watcher NativeHandle<*H>>(watcher: &r/mut W) -> &r/mut WatcherData {
    unsafe {
        let data = uvll::get_data_for_uv_handle(watcher.native_handle());
        let data = transmute::<&*c_void, &mut ~WatcherData>(&data);
        return &mut **data;
    }
}

fn drop_watcher_data<H, W: Watcher NativeHandle<*H>>(watcher: &mut W) {
    unsafe {
        let data = uvll::get_data_for_uv_handle(watcher.native_handle());
        let _data = transmute::<*c_void, ~WatcherData>(data);
        uvll::set_data_for_uv_handle(watcher.native_handle(), null::<()>());
    }
}

#[test]
fn test_slice_to_uv_buf() {
    let slice = [0, .. 20];
    let buf = slice_to_uv_buf(slice);

    assert buf.len == 20;

    unsafe {
        let base = transmute::<*u8, *mut u8>(buf.base);
        (*base) = 1;
        (*ptr::mut_offset(base, 1)) = 2;
    }

    assert slice[0] == 1;
    assert slice[1] == 2;
}

/// The uv buffer type
type Buf = uvll::uv_buf_t;

/// Borrow a slice to a Buf
fn slice_to_uv_buf(v: &[u8]) -> Buf {
    let data = unsafe { vec::raw::to_ptr(v) };
    unsafe { uvll::buf_init(data, v.len()) }
}

// FIXME: Do these conversions without copying

/// Transmute an owned vector to a Buf
fn vec_to_uv_buf(v: ~[u8]) -> Buf {
    let data = unsafe { malloc(v.len() as size_t) } as *u8;
    assert data.is_not_null();
    do vec::as_imm_buf(v) |b, l| {
        let data = data as *mut u8;
        unsafe { ptr::copy_memory(data, b, l) }
    }
    let buf = unsafe { uvll::buf_init(data, v.len()) };
    return buf;
}

/// Transmute a Buf that was once a ~[u8] back to ~[u8]
fn vec_from_uv_buf(buf: Buf) -> Option<~[u8]> {
    if !(buf.len == 0 && buf.base.is_null()) {
        let v = unsafe { vec::from_buf(buf.base, buf.len as uint) };
        unsafe { free(buf.base as *c_void) };
        return Some(v);
    } else {
        // No buffer
        return None;
    }
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
        let mut idle_watcher = { IdleWatcher::new(&mut loop_) };
        idle_watcher.close();
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
            assert status.is_none();
            if unsafe { *count_ptr == 10 } {
                idle_watcher.stop();
                idle_watcher.close();
            } else {
                unsafe { *count_ptr = *count_ptr + 1; }
            }
        }
        loop_.run();
        loop_.close();
        assert count == 10;
    }
}

#[test]
fn idle_start_stop_start() {
    do run_in_bare_thread {
        let mut loop_ = Loop::new();
        let mut idle_watcher = { IdleWatcher::new(&mut loop_) };
        do idle_watcher.start |idle_watcher, status| {
            let mut idle_watcher = idle_watcher;
            assert status.is_none();
            idle_watcher.stop();
            do idle_watcher.start |idle_watcher, status| {
                assert status.is_none();
                let mut idle_watcher = idle_watcher;
                idle_watcher.stop();
                idle_watcher.close();
            }
        }
        loop_.run();
        loop_.close();
    }
}

#[test]
fn connect_close() {
    do run_in_bare_thread() {
        let mut loop_ = Loop::new();
        let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
        // Connect to a port where nobody is listening
        let addr = ip4addr("127.0.0.1", 2923);
        do tcp_watcher.connect(addr) |stream_watcher, status| {
            rtdebug!("tcp_watcher.connect!");
            assert status.is_some();
            assert status.get().name() == ~"ECONNREFUSED";
            stream_watcher.close(||());
        }
        loop_.run();
        loop_.close();
    }
}

#[test]
#[ignore(reason = "need a server to connect to")]
fn connect_read() {
    do run_in_bare_thread() {
        let mut loop_ = Loop::new();
        let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
        let addr = ip4addr("127.0.0.1", 2924);
        do tcp_watcher.connect(addr) |stream_watcher, status| {
            let mut stream_watcher = stream_watcher;
            rtdebug!("tcp_watcher.connect!");
            assert status.is_none();
            let alloc: AllocCallback = |size| vec_to_uv_buf(vec::from_elem(size, 0));
            do stream_watcher.read_start(alloc) |stream_watcher, nread, buf, status| {
                let buf = vec_from_uv_buf(buf);
                rtdebug!("read cb!");
                if status.is_none() {
                    let bytes = buf.unwrap();
                    rtdebug!("%s", bytes.slice(0, nread as uint).to_str());
                } else {
                    rtdebug!("status after read: %s", status.get().to_str());
                    rtdebug!("closing");
                    stream_watcher.close(||());
                }
            }
        }
        loop_.run();
        loop_.close();
    }
}

#[test]
fn listen() {
    do run_in_bare_thread() {
        const MAX: int = 10;
        let mut loop_ = Loop::new();
        let mut server_tcp_watcher = { TcpWatcher::new(&mut loop_) };
        let addr = ip4addr("127.0.0.1", 2925);
        server_tcp_watcher.bind(addr);
        let loop_ = loop_;
        rtdebug!("listening");
        do server_tcp_watcher.listen |server_stream_watcher, status| {
            rtdebug!("listened!");
            assert status.is_none();
            let mut server_stream_watcher = server_stream_watcher;
            let mut loop_ = loop_;
            let mut client_tcp_watcher = TcpWatcher::new(&mut loop_);
            let mut client_tcp_watcher = client_tcp_watcher.as_stream();
            server_stream_watcher.accept(client_tcp_watcher);
            let count_cell = Cell(0);
            let server_stream_watcher = server_stream_watcher;
            rtdebug!("starting read");
            let alloc: AllocCallback = |size| vec_to_uv_buf(vec::from_elem(size, 0));
            do client_tcp_watcher.read_start(alloc) |stream_watcher, nread, buf, status| {
                rtdebug!("i'm reading!");
                let buf = vec_from_uv_buf(buf);
                let mut count = count_cell.take();
                if status.is_none() {
                    rtdebug!("got %d bytes", nread);
                    let buf = buf.unwrap();
                    for buf.view(0, nread as uint).each |byte| {
                        assert *byte == count as u8;
                        rtdebug!("%u", *byte as uint);
                        count += 1;
                    }
                } else {
                    assert count == MAX;
                    do stream_watcher.close {
                        server_stream_watcher.close(||());
                    }
                }
                count_cell.put_back(count);
            }
        }

        let _client_thread = do Thread::start {
            rtdebug!("starting client thread");
            let mut loop_ = Loop::new();
            let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
            do tcp_watcher.connect(addr) |stream_watcher, status| {
                rtdebug!("connecting");
                assert status.is_none();
                let mut stream_watcher = stream_watcher;
                let msg = ~[0, 1, 2, 3, 4, 5, 6 ,7 ,8, 9];
                do stream_watcher.write(msg) |stream_watcher, status| {
                    rtdebug!("writing");
                    assert status.is_none();
                    stream_watcher.close(||());
                }
            }
            loop_.run();
            loop_.close();
        };

        let mut loop_ = loop_;
        loop_.run();
        loop_.close();
    }
}
