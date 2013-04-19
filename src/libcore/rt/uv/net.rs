// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use libc::{size_t, ssize_t, c_int, c_void};
use cast::transmute_mut_region;
use super::super::uvll;
use super::super::uvll::*;
use super::{Loop, Watcher, Request, UvError, Buf, Callback, NativeHandle, NullCallback,
            loop_from_watcher, status_to_maybe_uv_error,
            install_watcher_data, get_watcher_data, drop_watcher_data,
            vec_to_uv_buf, vec_from_uv_buf};
use super::super::io::net::ip::{IpAddr, Ipv4, Ipv6};

#[cfg(test)]
use unstable::run_in_bare_thread;
#[cfg(test)]
use super::super::thread::Thread;
#[cfg(test)]
use cell::Cell;

fn ip4_as_uv_ip4(addr: IpAddr, f: &fn(*sockaddr_in)) {
    match addr {
        Ipv4(a, b, c, d, p) => {
            unsafe {
                let addr = malloc_ip4_addr(fmt!("%u.%u.%u.%u",
                                                a as uint,
                                                b as uint,
                                                c as uint,
                                                d as uint), p as int);
                do (|| {
                    f(addr);
                }).finally {
                    free_ip4_addr(addr);
                }
            }
        }
        Ipv6 => fail!()
    }
}

// uv_stream t is the parent class of uv_tcp_t, uv_pipe_t, uv_tty_t
// and uv_file_t
pub struct StreamWatcher(*uvll::uv_stream_t);

impl Watcher for StreamWatcher {
    fn event_loop(&self) -> Loop {
        loop_from_watcher(self)
    }
}

pub type ReadCallback = ~fn(StreamWatcher, int, Buf, Option<UvError>);
impl Callback for ReadCallback { }

// XXX: The uv alloc callback also has a *uv_handle_t arg
pub type AllocCallback = ~fn(uint) -> Buf;
impl Callback for AllocCallback { }

pub impl StreamWatcher {

    fn read_start(&mut self, alloc: AllocCallback, cb: ReadCallback) {
        // XXX: Borrowchk problems
        let data = get_watcher_data(unsafe { transmute_mut_region(self) });
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

        extern fn read_cb(stream: *uvll::uv_stream_t, nread: ssize_t, buf: Buf) {
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
        // It would be nice to drop the alloc and read callbacks here,
        // but read_stop may be called from inside one of them and we
        // would end up freeing the in-use environment
        let handle = self.native_handle();
        unsafe { uvll::read_stop(handle); }
    }

    // XXX: Needs to take &[u8], not ~[u8]
    fn write(&mut self, msg: ~[u8], cb: ConnectionCallback) {
        // XXX: Borrowck
        let data = get_watcher_data(unsafe { transmute_mut_region(self) });
        assert!(data.write_cb.is_none());
        data.write_cb = Some(cb);

        let req = WriteRequest::new();
        let buf = vec_to_uv_buf(msg);
        // XXX: Allocation
        let bufs = ~[buf];
        unsafe {
            assert!(0 == uvll::write(req.native_handle(),
                                          self.native_handle(),
                                          &bufs, write_cb));
        }
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
        unsafe {
            assert!(0 == uvll::accept(self_handle, stream_handle));
        }
    }

    fn close(self, cb: NullCallback) {
        {
            let mut self = self;
            let data = get_watcher_data(&mut self);
            assert!(data.close_cb.is_none());
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
            unsafe { free_handle(handle as *c_void) }
        }
    }
}

impl NativeHandle<*uvll::uv_stream_t> for StreamWatcher {
    fn from_native_handle(
        handle: *uvll::uv_stream_t) -> StreamWatcher {
        StreamWatcher(handle)
    }
    fn native_handle(&self) -> *uvll::uv_stream_t {
        match self { &StreamWatcher(ptr) => ptr }
    }
}

pub struct TcpWatcher(*uvll::uv_tcp_t);

impl Watcher for TcpWatcher {
    fn event_loop(&self) -> Loop {
        loop_from_watcher(self)
    }
}

pub type ConnectionCallback = ~fn(StreamWatcher, Option<UvError>);
impl Callback for ConnectionCallback { }

pub impl TcpWatcher {
    fn new(loop_: &mut Loop) -> TcpWatcher {
        unsafe {
            let handle = malloc_handle(UV_TCP);
            assert!(handle.is_not_null());
            assert!(0 == uvll::tcp_init(loop_.native_handle(), handle));
            let mut watcher = NativeHandle::from_native_handle(handle);
            install_watcher_data(&mut watcher);
            return watcher;
        }
    }

    fn bind(&mut self, address: IpAddr) {
        match address {
            Ipv4(*) => {
                do ip4_as_uv_ip4(address) |addr| {
                    let result = unsafe {
                        uvll::tcp_bind(self.native_handle(), addr)
                    };
                    // XXX: bind is likely to fail. need real error handling
                    assert!(result == 0);
                }
            }
            _ => fail!()
        }
    }

    fn connect(&mut self, address: IpAddr, cb: ConnectionCallback) {
        unsafe {
            assert!(get_watcher_data(self).connect_cb.is_none());
            get_watcher_data(self).connect_cb = Some(cb);

            let mut connect_watcher = ConnectRequest::new();
            let connect_handle = connect_watcher.native_handle();
            match address {
                Ipv4(*) => {
                    do ip4_as_uv_ip4(address) |addr| {
                        rtdebug!("connect_t: %x", connect_handle as uint);
                        assert!(0 == uvll::tcp_connect(connect_handle,
                                                            self.native_handle(),
                                                            addr, connect_cb));
                    }
                }
                _ => fail!()
            }

            extern fn connect_cb(req: *uvll::uv_connect_t, status: c_int) {
                rtdebug!("connect_t: %x", req as uint);
                let connect_request: ConnectRequest = NativeHandle::from_native_handle(req);
                let mut stream_watcher = connect_request.stream();
                connect_request.delete();
                let cb: ConnectionCallback = {
                    let data = get_watcher_data(&mut stream_watcher);
                    data.connect_cb.swap_unwrap()
                };
                let status = status_to_maybe_uv_error(stream_watcher.native_handle(), status);
                cb(stream_watcher, status);
            }
        }
    }

    fn listen(&mut self, cb: ConnectionCallback) {
        // XXX: Borrowck
        let data = get_watcher_data(unsafe { transmute_mut_region(self) });
        assert!(data.connect_cb.is_none());
        data.connect_cb = Some(cb);

        unsafe {
            static BACKLOG: c_int = 128; // XXX should be configurable
            // XXX: This can probably fail
            assert!(0 == uvll::listen(self.native_handle(),
                                           BACKLOG, connection_cb));
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

impl NativeHandle<*uvll::uv_tcp_t> for TcpWatcher {
    fn from_native_handle(handle: *uvll::uv_tcp_t) -> TcpWatcher {
        TcpWatcher(handle)
    }
    fn native_handle(&self) -> *uvll::uv_tcp_t {
        match self { &TcpWatcher(ptr) => ptr }
    }
}

pub type ConnectCallback = ~fn(ConnectRequest, Option<UvError>);
impl Callback for ConnectCallback { }

// uv_connect_t is a subclass of uv_req_t
struct ConnectRequest(*uvll::uv_connect_t);

impl Request for ConnectRequest { }

impl ConnectRequest {

    fn new() -> ConnectRequest {
        let connect_handle = unsafe {
            malloc_req(UV_CONNECT)
        };
        assert!(connect_handle.is_not_null());
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
        unsafe { free_req(self.native_handle() as *c_void) }
    }
}

impl NativeHandle<*uvll::uv_connect_t> for ConnectRequest {
    fn from_native_handle(
        handle: *uvll:: uv_connect_t) -> ConnectRequest {
        ConnectRequest(handle)
    }
    fn native_handle(&self) -> *uvll::uv_connect_t {
        match self { &ConnectRequest(ptr) => ptr }
    }
}

pub struct WriteRequest(*uvll::uv_write_t);

impl Request for WriteRequest { }

pub impl WriteRequest {

    fn new() -> WriteRequest {
        let write_handle = unsafe {
            malloc_req(UV_WRITE)
        };
        assert!(write_handle.is_not_null());
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
        unsafe { free_req(self.native_handle() as *c_void) }
    }
}

impl NativeHandle<*uvll::uv_write_t> for WriteRequest {
    fn from_native_handle(handle: *uvll:: uv_write_t) -> WriteRequest {
        WriteRequest(handle)
    }
    fn native_handle(&self) -> *uvll::uv_write_t {
        match self { &WriteRequest(ptr) => ptr }
    }
}


#[test]
fn connect_close() {
    do run_in_bare_thread() {
        let mut loop_ = Loop::new();
        let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
        // Connect to a port where nobody is listening
        let addr = Ipv4(127, 0, 0, 1, 2923);
        do tcp_watcher.connect(addr) |stream_watcher, status| {
            rtdebug!("tcp_watcher.connect!");
            assert!(status.is_some());
            assert!(status.get().name() == ~"ECONNREFUSED");
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
        let addr = Ipv4(127, 0, 0, 1, 2924);
        do tcp_watcher.connect(addr) |stream_watcher, status| {
            let mut stream_watcher = stream_watcher;
            rtdebug!("tcp_watcher.connect!");
            assert!(status.is_none());
            let alloc: AllocCallback = |size| {
                vec_to_uv_buf(vec::from_elem(size, 0))
            };
            do stream_watcher.read_start(alloc)
                |stream_watcher, _nread, buf, status| {

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
        static MAX: int = 10;
        let mut loop_ = Loop::new();
        let mut server_tcp_watcher = { TcpWatcher::new(&mut loop_) };
        let addr = Ipv4(127, 0, 0, 1, 2925);
        server_tcp_watcher.bind(addr);
        let loop_ = loop_;
        rtdebug!("listening");
        do server_tcp_watcher.listen |server_stream_watcher, status| {
            rtdebug!("listened!");
            assert!(status.is_none());
            let mut server_stream_watcher = server_stream_watcher;
            let mut loop_ = loop_;
            let mut client_tcp_watcher = TcpWatcher::new(&mut loop_);
            let mut client_tcp_watcher = client_tcp_watcher.as_stream();
            server_stream_watcher.accept(client_tcp_watcher);
            let count_cell = Cell(0);
            let server_stream_watcher = server_stream_watcher;
            rtdebug!("starting read");
            let alloc: AllocCallback = |size| {
                vec_to_uv_buf(vec::from_elem(size, 0))
            };
            do client_tcp_watcher.read_start(alloc)
                |stream_watcher, nread, buf, status| {

                rtdebug!("i'm reading!");
                let buf = vec_from_uv_buf(buf);
                let mut count = count_cell.take();
                if status.is_none() {
                    rtdebug!("got %d bytes", nread);
                    let buf = buf.unwrap();
                    for buf.slice(0, nread as uint).each |byte| {
                        assert!(*byte == count as u8);
                        rtdebug!("%u", *byte as uint);
                        count += 1;
                    }
                } else {
                    assert!(count == MAX);
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
                assert!(status.is_none());
                let mut stream_watcher = stream_watcher;
                let msg = ~[0, 1, 2, 3, 4, 5, 6 ,7 ,8, 9];
                do stream_watcher.write(msg) |stream_watcher, status| {
                    rtdebug!("writing");
                    assert!(status.is_none());
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
