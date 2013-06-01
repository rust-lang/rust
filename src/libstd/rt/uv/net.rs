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
use rt::uv::uvll;
use rt::uv::uvll::*;
use rt::uv::{AllocCallback, ConnectionCallback, ReadCallback};
use rt::uv::{Loop, Watcher, Request, UvError, Buf, NativeHandle, NullCallback,
             status_to_maybe_uv_error};
use rt::io::net::ip::{IpAddr, Ipv4, Ipv6};
use rt::uv::last_uv_error;

fn ip4_as_uv_ip4<T>(addr: IpAddr, f: &fn(*sockaddr_in) -> T) -> T {
    match addr {
        Ipv4(a, b, c, d, p) => {
            unsafe {
                let addr = malloc_ip4_addr(fmt!("%u.%u.%u.%u",
                                                a as uint,
                                                b as uint,
                                                c as uint,
                                                d as uint), p as int);
                do (|| {
                    f(addr)
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
impl Watcher for StreamWatcher { }

impl StreamWatcher {
    pub fn read_start(&mut self, alloc: AllocCallback, cb: ReadCallback) {
        {
            let data = self.get_watcher_data();
            data.alloc_cb = Some(alloc);
            data.read_cb = Some(cb);
        }

        let handle = self.native_handle();
        unsafe { uvll::read_start(handle, alloc_cb, read_cb); }

        extern fn alloc_cb(stream: *uvll::uv_stream_t, suggested_size: size_t) -> Buf {
            let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(stream);
            let data = stream_watcher.get_watcher_data();
            let alloc_cb = data.alloc_cb.get_ref();
            return (*alloc_cb)(suggested_size as uint);
        }

        extern fn read_cb(stream: *uvll::uv_stream_t, nread: ssize_t, buf: Buf) {
            rtdebug!("buf addr: %x", buf.base as uint);
            rtdebug!("buf len: %d", buf.len as int);
            let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(stream);
            let data = stream_watcher.get_watcher_data();
            let cb = data.read_cb.get_ref();
            let status = status_to_maybe_uv_error(stream, nread as c_int);
            (*cb)(stream_watcher, nread as int, buf, status);
        }
    }

    pub fn read_stop(&mut self) {
        // It would be nice to drop the alloc and read callbacks here,
        // but read_stop may be called from inside one of them and we
        // would end up freeing the in-use environment
        let handle = self.native_handle();
        unsafe { uvll::read_stop(handle); }
    }

    pub fn write(&mut self, buf: Buf, cb: ConnectionCallback) {
        {
            let data = self.get_watcher_data();
            assert!(data.write_cb.is_none());
            data.write_cb = Some(cb);
        }

        let req = WriteRequest::new();
        let bufs = [buf];
        unsafe {
            assert!(0 == uvll::write(req.native_handle(),
                                     self.native_handle(),
                                     bufs, write_cb));
        }

        extern fn write_cb(req: *uvll::uv_write_t, status: c_int) {
            let write_request: WriteRequest = NativeHandle::from_native_handle(req);
            let mut stream_watcher = write_request.stream();
            write_request.delete();
            let cb = {
                let data = stream_watcher.get_watcher_data();
                let cb = data.write_cb.swap_unwrap();
                cb
            };
            let status = status_to_maybe_uv_error(stream_watcher.native_handle(), status);
            cb(stream_watcher, status);
        }
    }

    pub fn accept(&mut self, stream: StreamWatcher) {
        let self_handle = self.native_handle() as *c_void;
        let stream_handle = stream.native_handle() as *c_void;
        unsafe {
            assert_eq!(0, uvll::accept(self_handle, stream_handle));
        }
    }

    pub fn close(self, cb: NullCallback) {
        {
            let mut this = self;
            let data = this.get_watcher_data();
            assert!(data.close_cb.is_none());
            data.close_cb = Some(cb);
        }

        unsafe { uvll::close(self.native_handle(), close_cb); }

        extern fn close_cb(handle: *uvll::uv_stream_t) {
            let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(handle);
            {
                let data = stream_watcher.get_watcher_data();
                data.close_cb.swap_unwrap()();
            }
            stream_watcher.drop_watcher_data();
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
impl Watcher for TcpWatcher { }

impl TcpWatcher {
    pub fn new(loop_: &mut Loop) -> TcpWatcher {
        unsafe {
            let handle = malloc_handle(UV_TCP);
            assert!(handle.is_not_null());
            assert_eq!(0, uvll::tcp_init(loop_.native_handle(), handle));
            let mut watcher: TcpWatcher = NativeHandle::from_native_handle(handle);
            watcher.install_watcher_data();
            return watcher;
        }
    }

    pub fn bind(&mut self, address: IpAddr) -> Result<(), UvError> {
        match address {
            Ipv4(*) => {
                do ip4_as_uv_ip4(address) |addr| {
                    let result = unsafe {
                        uvll::tcp_bind(self.native_handle(), addr)
                    };
                    if result == 0 {
                        Ok(())
                    } else {
                        Err(last_uv_error(self))
                    }
                }
            }
            _ => fail!()
        }
    }

    pub fn connect(&mut self, address: IpAddr, cb: ConnectionCallback) {
        unsafe {
            assert!(self.get_watcher_data().connect_cb.is_none());
            self.get_watcher_data().connect_cb = Some(cb);

            let connect_handle = ConnectRequest::new().native_handle();
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
                    let data = stream_watcher.get_watcher_data();
                    data.connect_cb.swap_unwrap()
                };
                let status = status_to_maybe_uv_error(stream_watcher.native_handle(), status);
                cb(stream_watcher, status);
            }
        }
    }

    pub fn listen(&mut self, cb: ConnectionCallback) {
        {
            let data = self.get_watcher_data();
            assert!(data.connect_cb.is_none());
            data.connect_cb = Some(cb);
        }

        unsafe {
            static BACKLOG: c_int = 128; // XXX should be configurable
            // XXX: This can probably fail
            assert!(0 == uvll::listen(self.native_handle(),
                                           BACKLOG, connection_cb));
        }

        extern fn connection_cb(handle: *uvll::uv_stream_t, status: c_int) {
            rtdebug!("connection_cb");
            let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(handle);
            let data = stream_watcher.get_watcher_data();
            let cb = data.connect_cb.get_ref();
            let status = status_to_maybe_uv_error(handle, status);
            (*cb)(stream_watcher, status);
        }
    }

    pub fn as_stream(&self) -> StreamWatcher {
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

impl WriteRequest {
    pub fn new() -> WriteRequest {
        let write_handle = unsafe {
            malloc_req(UV_WRITE)
        };
        assert!(write_handle.is_not_null());
        let write_handle = write_handle as *uvll::uv_write_t;
        WriteRequest(write_handle)
    }

    pub fn stream(&self) -> StreamWatcher {
        unsafe {
            let stream_handle = uvll::get_stream_handle_from_write_req(self.native_handle());
            NativeHandle::from_native_handle(stream_handle)
        }
    }

    pub fn delete(self) {
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


#[cfg(test)]
mod test {
    use super::*;
    use util::ignore;
    use cell::Cell;
    use vec;
    use unstable::run_in_bare_thread;
    use rt::thread::Thread;
    use rt::test::*;
    use rt::uv::{Loop, AllocCallback};
    use rt::uv::{vec_from_uv_buf, vec_to_uv_buf, slice_to_uv_buf};

    #[test]
    fn connect_close() {
        do run_in_bare_thread() {
            let mut loop_ = Loop::new();
            let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
            // Connect to a port where nobody is listening
            let addr = next_test_ip4();
            do tcp_watcher.connect(addr) |stream_watcher, status| {
                rtdebug!("tcp_watcher.connect!");
                assert!(status.is_some());
                assert_eq!(status.get().name(), ~"ECONNREFUSED");
                stream_watcher.close(||());
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
            let addr = next_test_ip4();
            server_tcp_watcher.bind(addr);
            let loop_ = loop_;
            rtdebug!("listening");
            do server_tcp_watcher.listen |server_stream_watcher, status| {
                rtdebug!("listened!");
                assert!(status.is_none());
                let mut server_stream_watcher = server_stream_watcher;
                let mut loop_ = loop_;
                let client_tcp_watcher = TcpWatcher::new(&mut loop_);
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
                        assert_eq!(count, MAX);
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
                    let buf = slice_to_uv_buf(msg);
                    let msg_cell = Cell(msg);
                    do stream_watcher.write(buf) |stream_watcher, status| {
                        rtdebug!("writing");
                        assert!(status.is_none());
                        let msg_cell = Cell(msg_cell.take());
                        stream_watcher.close(||ignore(msg_cell.take()));
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
}
