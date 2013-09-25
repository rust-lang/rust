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
use libc::{size_t, ssize_t, c_int, c_void, c_uint};
use rt::uv::uvll;
use rt::uv::uvll::*;
use rt::uv::{AllocCallback, ConnectionCallback, ReadCallback, UdpReceiveCallback, UdpSendCallback};
use rt::uv::{Loop, Watcher, Request, UvError, Buf, NativeHandle, NullCallback,
             status_to_maybe_uv_error};
use rt::io::net::ip::{SocketAddr, Ipv4Addr, Ipv6Addr};
use vec;
use str;
use from_str::{FromStr};

pub struct UvAddrInfo(*uvll::addrinfo);

pub enum UvSocketAddr {
    UvIpv4SocketAddr(*sockaddr_in),
    UvIpv6SocketAddr(*sockaddr_in6),
}

fn sockaddr_to_UvSocketAddr(addr: *uvll::sockaddr) -> UvSocketAddr {
    unsafe {
        assert!((is_ip4_addr(addr) || is_ip6_addr(addr)));
        assert!(!(is_ip4_addr(addr) && is_ip6_addr(addr)));
        match addr {
            _ if is_ip4_addr(addr) => UvIpv4SocketAddr(addr as *uvll::sockaddr_in),
            _ if is_ip6_addr(addr) => UvIpv6SocketAddr(addr as *uvll::sockaddr_in6),
            _ => fail!(),
        }
    }
}

fn socket_addr_as_uv_socket_addr<T>(addr: SocketAddr, f: &fn(UvSocketAddr) -> T) -> T {
    let malloc = match addr.ip {
        Ipv4Addr(*) => malloc_ip4_addr,
        Ipv6Addr(*) => malloc_ip6_addr,
    };
    let wrap = match addr.ip {
        Ipv4Addr(*) => UvIpv4SocketAddr,
        Ipv6Addr(*) => UvIpv6SocketAddr,
    };
    let free = match addr.ip {
        Ipv4Addr(*) => free_ip4_addr,
        Ipv6Addr(*) => free_ip6_addr,
    };

    let addr = unsafe { malloc(addr.ip.to_str(), addr.port as int) };
    do (|| {
        f(wrap(addr))
    }).finally {
        unsafe { free(addr) };
    }
}

fn uv_socket_addr_as_socket_addr<T>(addr: UvSocketAddr, f: &fn(SocketAddr) -> T) -> T {
    let ip_size = match addr {
        UvIpv4SocketAddr(*) => 4/*groups of*/ * 3/*digits separated by*/ + 3/*periods*/,
        UvIpv6SocketAddr(*) => 8/*groups of*/ * 4/*hex digits separated by*/ + 7 /*colons*/,
    };
    let ip_name = {
        let buf = vec::from_elem(ip_size + 1 /*null terminated*/, 0u8);
        unsafe {
            let buf_ptr = vec::raw::to_ptr(buf);
            match addr {
                UvIpv4SocketAddr(addr) => uvll::ip4_name(addr, buf_ptr, ip_size as size_t),
                UvIpv6SocketAddr(addr) => uvll::ip6_name(addr, buf_ptr, ip_size as size_t),
            }
        };
        buf
    };
    let ip_port = unsafe {
        let port = match addr {
            UvIpv4SocketAddr(addr) => uvll::ip4_port(addr),
            UvIpv6SocketAddr(addr) => uvll::ip6_port(addr),
        };
        port as u16
    };
    let ip_str = str::from_utf8_slice(ip_name).trim_right_chars(&'\x00');
    let ip_addr = FromStr::from_str(ip_str).unwrap();

    // finally run the closure
    f(SocketAddr { ip: ip_addr, port: ip_port })
}

pub fn uv_socket_addr_to_socket_addr(addr: UvSocketAddr) -> SocketAddr {
    use util;
    uv_socket_addr_as_socket_addr(addr, util::id)
}

// Traverse the addrinfo linked list, producing a vector of Rust socket addresses
pub fn accum_sockaddrs(addr: &UvAddrInfo) -> ~[SocketAddr] {
    unsafe {
        let &UvAddrInfo(addr) = addr;
        let mut addr = addr;

        let mut addrs = ~[];
        loop {
            let uvaddr = sockaddr_to_UvSocketAddr((*addr).ai_addr);
            let rustaddr = uv_socket_addr_to_socket_addr(uvaddr);
            addrs.push(rustaddr);
            if (*addr).ai_next.is_not_null() {
                addr = (*addr).ai_next;
            } else {
                break;
            }
        }

        return addrs;
    }
}

#[cfg(test)]
#[test]
fn test_ip4_conversion() {
    use rt;
    let ip4 = rt::test::next_test_ip4();
    assert_eq!(ip4, socket_addr_as_uv_socket_addr(ip4, uv_socket_addr_to_socket_addr));
}

#[cfg(test)]
#[test]
fn test_ip6_conversion() {
    use rt;
    let ip6 = rt::test::next_test_ip6();
    assert_eq!(ip6, socket_addr_as_uv_socket_addr(ip6, uv_socket_addr_to_socket_addr));
}

// uv_stream_t is the parent class of uv_tcp_t, uv_pipe_t, uv_tty_t
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

        unsafe { uvll::read_start(self.native_handle(), alloc_cb, read_cb); }

        extern fn alloc_cb(stream: *uvll::uv_stream_t, suggested_size: size_t) -> Buf {
            let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(stream);
            let alloc_cb = stream_watcher.get_watcher_data().alloc_cb.get_ref();
            return (*alloc_cb)(suggested_size as uint);
        }

        extern fn read_cb(stream: *uvll::uv_stream_t, nread: ssize_t, buf: Buf) {
            rtdebug!("buf addr: %x", buf.base as uint);
            rtdebug!("buf len: %d", buf.len as int);
            let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(stream);
            let cb = stream_watcher.get_watcher_data().read_cb.get_ref();
            let status = status_to_maybe_uv_error(nread as c_int);
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
        unsafe {
        assert_eq!(0, uvll::write(req.native_handle(), self.native_handle(), [buf], write_cb));
        }

        extern fn write_cb(req: *uvll::uv_write_t, status: c_int) {
            let write_request: WriteRequest = NativeHandle::from_native_handle(req);
            let mut stream_watcher = write_request.stream();
            write_request.delete();
            let cb = stream_watcher.get_watcher_data().write_cb.take_unwrap();
            let status = status_to_maybe_uv_error(status);
            cb(stream_watcher, status);
        }
    }

    pub fn accept(&mut self, stream: StreamWatcher) {
        let self_handle = self.native_handle() as *c_void;
        let stream_handle = stream.native_handle() as *c_void;
        assert_eq!(0, unsafe { uvll::accept(self_handle, stream_handle) } );
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
            let cb = stream_watcher.get_watcher_data().close_cb.take_unwrap();
            stream_watcher.drop_watcher_data();
            unsafe { free_handle(handle as *c_void) }
            cb();
        }
    }
}

impl NativeHandle<*uvll::uv_stream_t> for StreamWatcher {
    fn from_native_handle(handle: *uvll::uv_stream_t) -> StreamWatcher {
        StreamWatcher(handle)
    }
    fn native_handle(&self) -> *uvll::uv_stream_t {
        match self { &StreamWatcher(ptr) => ptr }
    }
}

pub struct TcpWatcher(*uvll::uv_tcp_t);
impl Watcher for TcpWatcher { }

impl TcpWatcher {
    pub fn new(loop_: &Loop) -> TcpWatcher {
        unsafe {
            let handle = malloc_handle(UV_TCP);
            assert!(handle.is_not_null());
            assert_eq!(0, uvll::tcp_init(loop_.native_handle(), handle));
            let mut watcher: TcpWatcher = NativeHandle::from_native_handle(handle);
            watcher.install_watcher_data();
            return watcher;
        }
    }

    pub fn bind(&mut self, address: SocketAddr) -> Result<(), UvError> {
        do socket_addr_as_uv_socket_addr(address) |addr| {
            let result = unsafe {
                match addr {
                    UvIpv4SocketAddr(addr) => uvll::tcp_bind(self.native_handle(), addr),
                    UvIpv6SocketAddr(addr) => uvll::tcp_bind6(self.native_handle(), addr),
                }
            };
            match result {
                0 => Ok(()),
                _ => Err(UvError(result)),
            }
        }
    }

    pub fn connect(&mut self, address: SocketAddr, cb: ConnectionCallback) {
        unsafe {
            assert!(self.get_watcher_data().connect_cb.is_none());
            self.get_watcher_data().connect_cb = Some(cb);

            let connect_handle = ConnectRequest::new().native_handle();
            rtdebug!("connect_t: %x", connect_handle as uint);
            do socket_addr_as_uv_socket_addr(address) |addr| {
                let result = match addr {
                    UvIpv4SocketAddr(addr) => uvll::tcp_connect(connect_handle,
                                                      self.native_handle(), addr, connect_cb),
                    UvIpv6SocketAddr(addr) => uvll::tcp_connect6(connect_handle,
                                                       self.native_handle(), addr, connect_cb),
                };
                assert_eq!(0, result);
            }

            extern fn connect_cb(req: *uvll::uv_connect_t, status: c_int) {
                rtdebug!("connect_t: %x", req as uint);
                let connect_request: ConnectRequest = NativeHandle::from_native_handle(req);
                let mut stream_watcher = connect_request.stream();
                connect_request.delete();
                let cb = stream_watcher.get_watcher_data().connect_cb.take_unwrap();
                let status = status_to_maybe_uv_error(status);
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
            assert_eq!(0, uvll::listen(self.native_handle(), BACKLOG, connection_cb));
        }

        extern fn connection_cb(handle: *uvll::uv_stream_t, status: c_int) {
            rtdebug!("connection_cb");
            let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(handle);
            let cb = stream_watcher.get_watcher_data().connect_cb.get_ref();
            let status = status_to_maybe_uv_error(status);
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

pub struct UdpWatcher(*uvll::uv_udp_t);
impl Watcher for UdpWatcher { }

impl UdpWatcher {
    pub fn new(loop_: &Loop) -> UdpWatcher {
        unsafe {
            let handle = malloc_handle(UV_UDP);
            assert!(handle.is_not_null());
            assert_eq!(0, uvll::udp_init(loop_.native_handle(), handle));
            let mut watcher: UdpWatcher = NativeHandle::from_native_handle(handle);
            watcher.install_watcher_data();
            return watcher;
        }
    }

    pub fn bind(&mut self, address: SocketAddr) -> Result<(), UvError> {
        do socket_addr_as_uv_socket_addr(address) |addr| {
            let result = unsafe {
                match addr {
                    UvIpv4SocketAddr(addr) => uvll::udp_bind(self.native_handle(), addr, 0u32),
                    UvIpv6SocketAddr(addr) => uvll::udp_bind6(self.native_handle(), addr, 0u32),
                }
            };
            match result {
                0 => Ok(()),
                _ => Err(UvError(result)),
            }
        }
    }

    pub fn recv_start(&mut self, alloc: AllocCallback, cb: UdpReceiveCallback) {
        {
            let data = self.get_watcher_data();
            data.alloc_cb = Some(alloc);
            data.udp_recv_cb = Some(cb);
        }

        unsafe { uvll::udp_recv_start(self.native_handle(), alloc_cb, recv_cb); }

        extern fn alloc_cb(handle: *uvll::uv_udp_t, suggested_size: size_t) -> Buf {
            let mut udp_watcher: UdpWatcher = NativeHandle::from_native_handle(handle);
            let alloc_cb = udp_watcher.get_watcher_data().alloc_cb.get_ref();
            return (*alloc_cb)(suggested_size as uint);
        }

        extern fn recv_cb(handle: *uvll::uv_udp_t, nread: ssize_t, buf: Buf,
                          addr: *uvll::sockaddr, flags: c_uint) {
            // When there's no data to read the recv callback can be a no-op.
            // This can happen if read returns EAGAIN/EWOULDBLOCK. By ignoring
            // this we just drop back to kqueue and wait for the next callback.
            if nread == 0 {
                return;
            }

            rtdebug!("buf addr: %x", buf.base as uint);
            rtdebug!("buf len: %d", buf.len as int);
            let mut udp_watcher: UdpWatcher = NativeHandle::from_native_handle(handle);
            let cb = udp_watcher.get_watcher_data().udp_recv_cb.get_ref();
            let status = status_to_maybe_uv_error(nread as c_int);
            let addr = uv_socket_addr_to_socket_addr(sockaddr_to_UvSocketAddr(addr));
            (*cb)(udp_watcher, nread as int, buf, addr, flags as uint, status);
        }
    }

    pub fn recv_stop(&mut self) {
        unsafe { uvll::udp_recv_stop(self.native_handle()); }
    }

    pub fn send(&mut self, buf: Buf, address: SocketAddr, cb: UdpSendCallback) {
        {
            let data = self.get_watcher_data();
            assert!(data.udp_send_cb.is_none());
            data.udp_send_cb = Some(cb);
        }

        let req = UdpSendRequest::new();
        do socket_addr_as_uv_socket_addr(address) |addr| {
            let result = unsafe {
                match addr {
                    UvIpv4SocketAddr(addr) => uvll::udp_send(req.native_handle(),
                                                   self.native_handle(), [buf], addr, send_cb),
                    UvIpv6SocketAddr(addr) => uvll::udp_send6(req.native_handle(),
                                                    self.native_handle(), [buf], addr, send_cb),
                }
            };
            assert_eq!(0, result);
        }

        extern fn send_cb(req: *uvll::uv_udp_send_t, status: c_int) {
            let send_request: UdpSendRequest = NativeHandle::from_native_handle(req);
            let mut udp_watcher = send_request.handle();
            send_request.delete();
            let cb = udp_watcher.get_watcher_data().udp_send_cb.take_unwrap();
            let status = status_to_maybe_uv_error(status);
            cb(udp_watcher, status);
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

        extern fn close_cb(handle: *uvll::uv_udp_t) {
            let mut udp_watcher: UdpWatcher = NativeHandle::from_native_handle(handle);
            let cb = udp_watcher.get_watcher_data().close_cb.take_unwrap();
            udp_watcher.drop_watcher_data();
            unsafe { free_handle(handle as *c_void) }
            cb();
        }
    }
}

impl NativeHandle<*uvll::uv_udp_t> for UdpWatcher {
    fn from_native_handle(handle: *uvll::uv_udp_t) -> UdpWatcher {
        UdpWatcher(handle)
    }
    fn native_handle(&self) -> *uvll::uv_udp_t {
        match self { &UdpWatcher(ptr) => ptr }
    }
}

// uv_connect_t is a subclass of uv_req_t
struct ConnectRequest(*uvll::uv_connect_t);
impl Request for ConnectRequest { }

impl ConnectRequest {

    fn new() -> ConnectRequest {
        let connect_handle = unsafe { malloc_req(UV_CONNECT) };
        assert!(connect_handle.is_not_null());
        ConnectRequest(connect_handle as *uvll::uv_connect_t)
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
    fn from_native_handle(handle: *uvll:: uv_connect_t) -> ConnectRequest {
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
        let write_handle = unsafe { malloc_req(UV_WRITE) };
        assert!(write_handle.is_not_null());
        WriteRequest(write_handle as *uvll::uv_write_t)
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

pub struct UdpSendRequest(*uvll::uv_udp_send_t);
impl Request for UdpSendRequest { }

impl UdpSendRequest {
    pub fn new() -> UdpSendRequest {
        let send_handle = unsafe { malloc_req(UV_UDP_SEND) };
        assert!(send_handle.is_not_null());
        UdpSendRequest(send_handle as *uvll::uv_udp_send_t)
    }

    pub fn handle(&self) -> UdpWatcher {
        let send_request_handle = unsafe {
            uvll::get_udp_handle_from_send_req(self.native_handle())
        };
        NativeHandle::from_native_handle(send_request_handle)
    }

    pub fn delete(self) {
        unsafe { free_req(self.native_handle() as *c_void) }
    }
}

impl NativeHandle<*uvll::uv_udp_send_t> for UdpSendRequest {
    fn from_native_handle(handle: *uvll::uv_udp_send_t) -> UdpSendRequest {
        UdpSendRequest(handle)
    }
    fn native_handle(&self) -> *uvll::uv_udp_send_t {
        match self { &UdpSendRequest(ptr) => ptr }
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
    use prelude::*;

    #[test]
    fn connect_close_ip4() {
        do run_in_bare_thread() {
            let mut loop_ = Loop::new();
            let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
            // Connect to a port where nobody is listening
            let addr = next_test_ip4();
            do tcp_watcher.connect(addr) |stream_watcher, status| {
                rtdebug!("tcp_watcher.connect!");
                assert!(status.is_some());
                assert_eq!(status.unwrap().name(), ~"ECONNREFUSED");
                stream_watcher.close(||());
            }
            loop_.run();
            loop_.close();
        }
    }

    #[test]
    fn connect_close_ip6() {
        do run_in_bare_thread() {
            let mut loop_ = Loop::new();
            let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
            // Connect to a port where nobody is listening
            let addr = next_test_ip6();
            do tcp_watcher.connect(addr) |stream_watcher, status| {
                rtdebug!("tcp_watcher.connect!");
                assert!(status.is_some());
                assert_eq!(status.unwrap().name(), ~"ECONNREFUSED");
                stream_watcher.close(||());
            }
            loop_.run();
            loop_.close();
        }
    }

    #[test]
    fn udp_bind_close_ip4() {
        do run_in_bare_thread() {
            let mut loop_ = Loop::new();
            let mut udp_watcher = { UdpWatcher::new(&mut loop_) };
            let addr = next_test_ip4();
            udp_watcher.bind(addr);
            udp_watcher.close(||());
            loop_.run();
            loop_.close();
        }
    }

    #[test]
    fn udp_bind_close_ip6() {
        do run_in_bare_thread() {
            let mut loop_ = Loop::new();
            let mut udp_watcher = { UdpWatcher::new(&mut loop_) };
            let addr = next_test_ip6();
            udp_watcher.bind(addr);
            udp_watcher.close(||());
            loop_.run();
            loop_.close();
        }
    }

    #[test]
    fn listen_ip4() {
        do run_in_bare_thread() {
            static MAX: int = 10;
            let mut loop_ = Loop::new();
            let mut server_tcp_watcher = { TcpWatcher::new(&mut loop_) };
            let addr = next_test_ip4();
            server_tcp_watcher.bind(addr);
            let loop_ = loop_;
            rtdebug!("listening");
            do server_tcp_watcher.listen |mut server_stream_watcher, status| {
                rtdebug!("listened!");
                assert!(status.is_none());
                let mut loop_ = loop_;
                let client_tcp_watcher = TcpWatcher::new(&mut loop_);
                let mut client_tcp_watcher = client_tcp_watcher.as_stream();
                server_stream_watcher.accept(client_tcp_watcher);
                let count_cell = Cell::new(0);
                let server_stream_watcher = server_stream_watcher;
                rtdebug!("starting read");
                let alloc: AllocCallback = |size| {
                    vec_to_uv_buf(vec::from_elem(size, 0u8))
                };
                do client_tcp_watcher.read_start(alloc) |stream_watcher, nread, buf, status| {

                    rtdebug!("i'm reading!");
                    let buf = vec_from_uv_buf(buf);
                    let mut count = count_cell.take();
                    if status.is_none() {
                        rtdebug!("got %d bytes", nread);
                        let buf = buf.unwrap();
                        for byte in buf.slice(0, nread as uint).iter() {
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

            let client_thread = do Thread::start {
                rtdebug!("starting client thread");
                let mut loop_ = Loop::new();
                let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
                do tcp_watcher.connect(addr) |mut stream_watcher, status| {
                    rtdebug!("connecting");
                    assert!(status.is_none());
                    let msg = ~[0, 1, 2, 3, 4, 5, 6 ,7 ,8, 9];
                    let buf = slice_to_uv_buf(msg);
                    let msg_cell = Cell::new(msg);
                    do stream_watcher.write(buf) |stream_watcher, status| {
                        rtdebug!("writing");
                        assert!(status.is_none());
                        let msg_cell = Cell::new(msg_cell.take());
                        stream_watcher.close(||ignore(msg_cell.take()));
                    }
                }
                loop_.run();
                loop_.close();
            };

            let mut loop_ = loop_;
            loop_.run();
            loop_.close();
            client_thread.join();
        }
    }

    #[test]
    fn listen_ip6() {
        do run_in_bare_thread() {
            static MAX: int = 10;
            let mut loop_ = Loop::new();
            let mut server_tcp_watcher = { TcpWatcher::new(&mut loop_) };
            let addr = next_test_ip6();
            server_tcp_watcher.bind(addr);
            let loop_ = loop_;
            rtdebug!("listening");
            do server_tcp_watcher.listen |mut server_stream_watcher, status| {
                rtdebug!("listened!");
                assert!(status.is_none());
                let mut loop_ = loop_;
                let client_tcp_watcher = TcpWatcher::new(&mut loop_);
                let mut client_tcp_watcher = client_tcp_watcher.as_stream();
                server_stream_watcher.accept(client_tcp_watcher);
                let count_cell = Cell::new(0);
                let server_stream_watcher = server_stream_watcher;
                rtdebug!("starting read");
                let alloc: AllocCallback = |size| {
                    vec_to_uv_buf(vec::from_elem(size, 0u8))
                };
                do client_tcp_watcher.read_start(alloc)
                    |stream_watcher, nread, buf, status| {

                    rtdebug!("i'm reading!");
                    let buf = vec_from_uv_buf(buf);
                    let mut count = count_cell.take();
                    if status.is_none() {
                        rtdebug!("got %d bytes", nread);
                        let buf = buf.unwrap();
                        let r = buf.slice(0, nread as uint);
                        for byte in r.iter() {
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

            let client_thread = do Thread::start {
                rtdebug!("starting client thread");
                let mut loop_ = Loop::new();
                let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
                do tcp_watcher.connect(addr) |mut stream_watcher, status| {
                    rtdebug!("connecting");
                    assert!(status.is_none());
                    let msg = ~[0, 1, 2, 3, 4, 5, 6 ,7 ,8, 9];
                    let buf = slice_to_uv_buf(msg);
                    let msg_cell = Cell::new(msg);
                    do stream_watcher.write(buf) |stream_watcher, status| {
                        rtdebug!("writing");
                        assert!(status.is_none());
                        let msg_cell = Cell::new(msg_cell.take());
                        stream_watcher.close(||ignore(msg_cell.take()));
                    }
                }
                loop_.run();
                loop_.close();
            };

            let mut loop_ = loop_;
            loop_.run();
            loop_.close();
            client_thread.join();
        }
    }

    #[test]
    fn udp_recv_ip4() {
        do run_in_bare_thread() {
            static MAX: int = 10;
            let mut loop_ = Loop::new();
            let server_addr = next_test_ip4();
            let client_addr = next_test_ip4();

            let mut server = UdpWatcher::new(&loop_);
            assert!(server.bind(server_addr).is_ok());

            rtdebug!("starting read");
            let alloc: AllocCallback = |size| {
                vec_to_uv_buf(vec::from_elem(size, 0u8))
            };

            do server.recv_start(alloc) |mut server, nread, buf, src, flags, status| {
                server.recv_stop();
                rtdebug!("i'm reading!");
                assert!(status.is_none());
                assert_eq!(flags, 0);
                assert_eq!(src, client_addr);

                let buf = vec_from_uv_buf(buf);
                let mut count = 0;
                rtdebug!("got %d bytes", nread);

                let buf = buf.unwrap();
                for &byte in buf.slice(0, nread as uint).iter() {
                    assert!(byte == count as u8);
                    rtdebug!("%u", byte as uint);
                    count += 1;
                }
                assert_eq!(count, MAX);

                server.close(||{});
            }

            let thread = do Thread::start {
                let mut loop_ = Loop::new();
                let mut client = UdpWatcher::new(&loop_);
                assert!(client.bind(client_addr).is_ok());
                let msg = ~[0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
                let buf = slice_to_uv_buf(msg);
                do client.send(buf, server_addr) |client, status| {
                    rtdebug!("writing");
                    assert!(status.is_none());
                    client.close(||{});
                }

                loop_.run();
                loop_.close();
            };

            loop_.run();
            loop_.close();
            thread.join();
        }
    }

    #[test]
    fn udp_recv_ip6() {
        do run_in_bare_thread() {
            static MAX: int = 10;
            let mut loop_ = Loop::new();
            let server_addr = next_test_ip6();
            let client_addr = next_test_ip6();

            let mut server = UdpWatcher::new(&loop_);
            assert!(server.bind(server_addr).is_ok());

            rtdebug!("starting read");
            let alloc: AllocCallback = |size| {
                vec_to_uv_buf(vec::from_elem(size, 0u8))
            };

            do server.recv_start(alloc) |mut server, nread, buf, src, flags, status| {
                server.recv_stop();
                rtdebug!("i'm reading!");
                assert!(status.is_none());
                assert_eq!(flags, 0);
                assert_eq!(src, client_addr);

                let buf = vec_from_uv_buf(buf);
                let mut count = 0;
                rtdebug!("got %d bytes", nread);

                let buf = buf.unwrap();
                for &byte in buf.slice(0, nread as uint).iter() {
                    assert!(byte == count as u8);
                    rtdebug!("%u", byte as uint);
                    count += 1;
                }
                assert_eq!(count, MAX);

                server.close(||{});
            }

            let thread = do Thread::start {
                let mut loop_ = Loop::new();
                let mut client = UdpWatcher::new(&loop_);
                assert!(client.bind(client_addr).is_ok());
                let msg = ~[0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
                let buf = slice_to_uv_buf(msg);
                do client.send(buf, server_addr) |client, status| {
                    rtdebug!("writing");
                    assert!(status.is_none());
                    client.close(||{});
                }

                loop_.run();
                loop_.close();
            };

            loop_.run();
            loop_.close();
            thread.join();
        }
    }
}
