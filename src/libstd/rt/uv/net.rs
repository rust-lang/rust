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
use rt::io::net::ip::{IpAddr, Ipv4, Ipv6};
use rt::uv::last_uv_error;
use vec;
use str;
use from_str::{FromStr};
use num;

pub enum UvIpAddr {
    UvIpv4(*sockaddr_in),
    UvIpv6(*sockaddr_in6),
}

fn sockaddr_to_UvIpAddr(addr: *uvll::sockaddr) -> UvIpAddr {
    unsafe {
        assert!((is_ip4_addr(addr) || is_ip6_addr(addr)));
        assert!(!(is_ip4_addr(addr) && is_ip6_addr(addr)));
        match addr {
            _ if is_ip4_addr(addr) => UvIpv4(addr as *uvll::sockaddr_in),
            _ if is_ip6_addr(addr) => UvIpv6(addr as *uvll::sockaddr_in6),
            _ => fail!(),
        }
    }
}

fn ip_as_uv_ip<T>(addr: IpAddr, f: &fn(UvIpAddr) -> T) -> T {
    let malloc = match addr {
        Ipv4(*) => malloc_ip4_addr,
        Ipv6(*) => malloc_ip6_addr,
    };
    let wrap = match addr {
        Ipv4(*) => UvIpv4,
        Ipv6(*) => UvIpv6,
    };
    let ip_str = match addr {
        Ipv4(x1, x2, x3, x4, _) =>
            fmt!("%u.%u.%u.%u", x1 as uint, x2 as uint, x3 as uint, x4 as uint),
        Ipv6(x1, x2, x3, x4, x5, x6, x7, x8, _) =>
            fmt!("%x:%x:%x:%x:%x:%x:%x:%x",
                  x1 as uint, x2 as uint, x3 as uint, x4 as uint,
                  x5 as uint, x6 as uint, x7 as uint, x8 as uint),
    };
    let port = match addr {
        Ipv4(_, _, _, _, p) | Ipv6(_, _, _, _, _, _, _, _, p) => p as int
    };
    let free = match addr {
        Ipv4(*) => free_ip4_addr,
        Ipv6(*) => free_ip6_addr,
    };

    let addr = unsafe { malloc(ip_str, port) };
    do (|| {
        f(wrap(addr))
    }).finally {
        unsafe { free(addr) };
    }
}

fn uv_ip_as_ip<T>(addr: UvIpAddr, f: &fn(IpAddr) -> T) -> T {
    let ip_size = match addr {
        UvIpv4(*) => 4/*groups of*/ * 3/*digits separated by*/ + 3/*periods*/,
        UvIpv6(*) => 8/*groups of*/ * 4/*hex digits separated by*/ + 7 /*colons*/,
    };
    let ip_name = {
        let buf = vec::from_elem(ip_size + 1 /*null terminated*/, 0u8);
        unsafe {
            match addr {
                UvIpv4(addr) => uvll::ip4_name(addr, vec::raw::to_ptr(buf), ip_size as size_t),
                UvIpv6(addr) => uvll::ip6_name(addr, vec::raw::to_ptr(buf), ip_size as size_t),
            }
        };
        buf
    };
    let ip_port = unsafe {
        let port = match addr {
            UvIpv4(addr) => uvll::ip4_port(addr),
            UvIpv6(addr) => uvll::ip6_port(addr),
        };
        port as u16
    };
    let ip_str = str::from_bytes_slice(ip_name).trim_right_chars(&'\x00');
    let ip = match addr {
        UvIpv4(*) => {
            let ip: ~[u8] =
                ip_str.split_iter('.')
                      .transform(|s: &str| -> u8 { FromStr::from_str(s).unwrap() })
                      .collect();
            assert_eq!(ip.len(), 4);
            Ipv4(ip[0], ip[1], ip[2], ip[3], ip_port)
        },
        UvIpv6(*) => {
            let ip: ~[u16] = {
                let expand_shorthand_and_convert = |s: &str| -> ~[~[u16]] {
                    let convert_each_segment = |s: &str| -> ~[u16] {
                        let read_hex_segment = |s: &str| -> u16 {
                            num::FromStrRadix::from_str_radix(s, 16u).unwrap()
                        };
                        match s {
                            "" => ~[],
                            // IPv4-Mapped/Compatible IPv6 Address?
                            s if s.find('.').is_some() => {
                                let i = s.rfind(':').get_or_default(-1);

                                let b = s.slice(i + 1, s.len()); // the ipv4 part

                                let h = b.split_iter('.')
                                   .transform(|s: &str| -> u8 { FromStr::from_str(s).unwrap() })
                                   .transform(|s: u8| -> ~str { fmt!("%02x", s as uint) })
                                   .collect::<~[~str]>();

                                if i == -1 {
                                    // Ipv4 Compatible Address (::x.x.x.x)
                                    // first 96 bits are zero leaving 32 bits
                                    // for the ipv4 part
                                    // (i.e ::127.0.0.1 == ::7F00:1)
                                    ~[num::FromStrRadix::from_str_radix(h[0] + h[1], 16).unwrap(),
                                      num::FromStrRadix::from_str_radix(h[2] + h[3], 16).unwrap()]
                                } else {
                                    // Ipv4-Mapped Address (::FFFF:x.x.x.x)
                                    // first 80 bits are zero, followed by all ones
                                    // for the next 16 bits, leaving 32 bits for
                                    // the ipv4 part
                                    // (i.e ::FFFF:127.0.0.1 == ::FFFF:7F00:1)
                                    ~[1,
                                      num::FromStrRadix::from_str_radix(h[0] + h[1], 16).unwrap(),
                                      num::FromStrRadix::from_str_radix(h[2] + h[3], 16).unwrap()]
                                }
                            },
                            s => s.split_iter(':').transform(read_hex_segment).collect()
                        }
                    };
                    s.split_str_iter("::").transform(convert_each_segment).collect()
                };
                match expand_shorthand_and_convert(ip_str) {
                    [x] => x, // no shorthand found
                    [l, r] => l + vec::from_elem(8 - l.len() - r.len(), 0u16) + r, // fill the gap
                    _ => fail!(), // impossible. only one shorthand allowed.
                }
            };
            assert_eq!(ip.len(), 8);
            Ipv6(ip[0], ip[1], ip[2], ip[3], ip[4], ip[5], ip[6], ip[7], ip_port)
        },
    };

    // finally run the closure
    f(ip)
}

pub fn uv_ip_to_ip(addr: UvIpAddr) -> IpAddr {
    use util;
    uv_ip_as_ip(addr, util::id)
}

#[cfg(test)]
#[test]
fn test_ip4_conversion() {
    use rt;
    let ip4 = rt::test::next_test_ip4();
    assert_eq!(ip4, ip_as_uv_ip(ip4, uv_ip_to_ip));
}

#[cfg(test)]
#[test]
fn test_ip6_conversion() {
    use rt;
    let ip6 = rt::test::next_test_ip6();
    assert_eq!(ip6, ip_as_uv_ip(ip6, uv_ip_to_ip));
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
            let status = status_to_maybe_uv_error(stream_watcher, nread as c_int);
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
            let status = status_to_maybe_uv_error(stream_watcher, status);
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
            stream_watcher.get_watcher_data().close_cb.take_unwrap()();
            stream_watcher.drop_watcher_data();
            unsafe { free_handle(handle as *c_void) }
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

    pub fn bind(&mut self, address: IpAddr) -> Result<(), UvError> {
        do ip_as_uv_ip(address) |addr| {
            let result = unsafe {
                match addr {
                    UvIpv4(addr) => uvll::tcp_bind(self.native_handle(), addr),
                    UvIpv6(addr) => uvll::tcp_bind6(self.native_handle(), addr),
                }
            };
            match result {
                0 => Ok(()),
                _ => Err(last_uv_error(self)),
            }
        }
    }

    pub fn connect(&mut self, address: IpAddr, cb: ConnectionCallback) {
        unsafe {
            assert!(self.get_watcher_data().connect_cb.is_none());
            self.get_watcher_data().connect_cb = Some(cb);

            let connect_handle = ConnectRequest::new().native_handle();
            rtdebug!("connect_t: %x", connect_handle as uint);
            do ip_as_uv_ip(address) |addr| {
                let result = match addr {
                    UvIpv4(addr) => uvll::tcp_connect(connect_handle,
                                                      self.native_handle(), addr, connect_cb),
                    UvIpv6(addr) => uvll::tcp_connect6(connect_handle,
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
                let status = status_to_maybe_uv_error(stream_watcher, status);
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
            let status = status_to_maybe_uv_error(stream_watcher, status);
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

    pub fn bind(&mut self, address: IpAddr) -> Result<(), UvError> {
        do ip_as_uv_ip(address) |addr| {
            let result = unsafe {
                match addr {
                    UvIpv4(addr) => uvll::udp_bind(self.native_handle(), addr, 0u32),
                    UvIpv6(addr) => uvll::udp_bind6(self.native_handle(), addr, 0u32),
                }
            };
            match result {
                0 => Ok(()),
                _ => Err(last_uv_error(self)),
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
            let status = status_to_maybe_uv_error(udp_watcher, nread as c_int);
            let addr = uv_ip_to_ip(sockaddr_to_UvIpAddr(addr));
            (*cb)(udp_watcher, nread as int, buf, addr, flags as uint, status);
        }
    }

    pub fn recv_stop(&mut self) {
        unsafe { uvll::udp_recv_stop(self.native_handle()); }
    }

    pub fn send(&mut self, buf: Buf, address: IpAddr, cb: UdpSendCallback) {
        {
            let data = self.get_watcher_data();
            assert!(data.udp_send_cb.is_none());
            data.udp_send_cb = Some(cb);
        }

        let req = UdpSendRequest::new();
        do ip_as_uv_ip(address) |addr| {
            let result = unsafe {
                match addr {
                    UvIpv4(addr) => uvll::udp_send(req.native_handle(),
                                                   self.native_handle(), [buf], addr, send_cb),
                    UvIpv6(addr) => uvll::udp_send6(req.native_handle(),
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
            let status = status_to_maybe_uv_error(udp_watcher, status);
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
            udp_watcher.get_watcher_data().close_cb.take_unwrap()();
            udp_watcher.drop_watcher_data();
            unsafe { free_handle(handle as *c_void) }
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
                assert_eq!(status.get().name(), ~"ECONNREFUSED");
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
                assert_eq!(status.get().name(), ~"ECONNREFUSED");
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
                        foreach byte in buf.slice(0, nread as uint).iter() {
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
                        foreach byte in r.iter() {
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
                foreach &byte in buf.slice(0, nread as uint).iter() {
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
                foreach &byte in buf.slice(0, nread as uint).iter() {
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
