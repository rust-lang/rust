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
use std::libc;
use std::libc::{size_t, ssize_t, c_int, c_void, c_uint, c_char};
use std::ptr;
use std::rt::BlockedTask;
use std::io::IoError;
use std::io::net::ip::{Ipv4Addr, Ipv6Addr, SocketAddr, IpAddr};
use std::rt::local::Local;
use std::rt::rtio;
use std::rt::sched::{Scheduler, SchedHandle};
use std::rt::tube::Tube;
use std::str;
use std::vec;

use stream::StreamWatcher;
use super::{Loop, Request, UvError, Buf, status_to_io_result,
            uv_error_to_io_error, UvHandle, slice_to_uv_buf,
            wait_until_woken_after};
use uvio::HomingIO;
use uvll;
use uvll::sockaddr;

////////////////////////////////////////////////////////////////////////////////
/// Generic functions related to dealing with sockaddr things
////////////////////////////////////////////////////////////////////////////////

fn socket_addr_as_sockaddr<T>(addr: SocketAddr, f: |*sockaddr| -> T) -> T {
    let malloc = match addr.ip {
        Ipv4Addr(..) => uvll::rust_malloc_ip4_addr,
        Ipv6Addr(..) => uvll::rust_malloc_ip6_addr,
    };

    let ip = addr.ip.to_str();
    let addr = ip.with_c_str(|p| unsafe { malloc(p, addr.port as c_int) });
    (|| {
        f(addr)
    }).finally(|| {
        unsafe { libc::free(addr) };
    })
}

pub fn sockaddr_to_socket_addr(addr: *sockaddr) -> SocketAddr {
    unsafe {
        let ip_size = if uvll::rust_is_ipv4_sockaddr(addr) == 1 {
            4/*groups of*/ * 3/*digits separated by*/ + 3/*periods*/
        } else if uvll::rust_is_ipv6_sockaddr(addr) == 1 {
            8/*groups of*/ * 4/*hex digits separated by*/ + 7 /*colons*/
        } else {
            fail!("unknown address?");
        };
        let ip_name = {
            // apparently there's an off-by-one in libuv?
            let ip_size = ip_size + 1;
            let buf = vec::from_elem(ip_size + 1 /*null terminated*/, 0u8);
            let buf_ptr = vec::raw::to_ptr(buf);
            let ret = if uvll::rust_is_ipv4_sockaddr(addr) == 1 {
                uvll::uv_ip4_name(addr, buf_ptr as *c_char, ip_size as size_t)
            } else {
                uvll::uv_ip6_name(addr, buf_ptr as *c_char, ip_size as size_t)
            };
            if ret != 0 {
                fail!("error parsing sockaddr: {}", UvError(ret).desc());
            }
            buf
        };
        let ip_port = {
            let port = if uvll::rust_is_ipv4_sockaddr(addr) == 1 {
                uvll::rust_ip4_port(addr)
            } else {
                uvll::rust_ip6_port(addr)
            };
            port as u16
        };
        let ip_str = str::from_utf8(ip_name).trim_right_chars(&'\x00');
        let ip_addr = FromStr::from_str(ip_str).unwrap();

        SocketAddr { ip: ip_addr, port: ip_port }
    }
}

#[cfg(test)]
#[test]
fn test_ip4_conversion() {
    use std::rt;
    let ip4 = rt::test::next_test_ip4();
    socket_addr_as_sockaddr(ip4, |addr| {
        assert_eq!(ip4, sockaddr_to_socket_addr(addr));
    })
}

#[cfg(test)]
#[test]
fn test_ip6_conversion() {
    use std::rt;
    let ip6 = rt::test::next_test_ip6();
    socket_addr_as_sockaddr(ip6, |addr| {
        assert_eq!(ip6, sockaddr_to_socket_addr(addr));
    })
}

enum SocketNameKind {
    TcpPeer,
    Tcp,
    Udp
}

fn socket_name(sk: SocketNameKind, handle: *c_void) -> Result<SocketAddr, IoError> {
    unsafe {
        let getsockname = match sk {
            TcpPeer => uvll::uv_tcp_getpeername,
            Tcp     => uvll::uv_tcp_getsockname,
            Udp     => uvll::uv_udp_getsockname,
        };

        // Allocate a sockaddr_storage
        // since we don't know if it's ipv4 or ipv6
        let size = uvll::rust_sockaddr_size();
        let name = libc::malloc(size as size_t);
        assert!(!name.is_null());
        let mut namelen = size;

        let ret = match getsockname(handle, name, &mut namelen) {
            0 => Ok(sockaddr_to_socket_addr(name)),
            n => Err(uv_error_to_io_error(UvError(n)))
        };
        libc::free(name);
        ret
    }
}

////////////////////////////////////////////////////////////////////////////////
/// TCP implementation
////////////////////////////////////////////////////////////////////////////////

pub struct TcpWatcher {
    handle: *uvll::uv_tcp_t,
    stream: StreamWatcher,
    home: SchedHandle,
}

pub struct TcpListener {
    home: SchedHandle,
    handle: *uvll::uv_pipe_t,
    priv closing_task: Option<BlockedTask>,
    priv outgoing: Tube<Result<~rtio::RtioTcpStream, IoError>>,
}

pub struct TcpAcceptor {
    listener: ~TcpListener,
    priv incoming: Tube<Result<~rtio::RtioTcpStream, IoError>>,
}

// TCP watchers (clients/streams)

impl TcpWatcher {
    pub fn new(loop_: &Loop) -> TcpWatcher {
        let handle = unsafe { uvll::malloc_handle(uvll::UV_TCP) };
        assert_eq!(unsafe {
            uvll::uv_tcp_init(loop_.handle, handle)
        }, 0);
        TcpWatcher {
            home: get_handle_to_current_scheduler!(),
            handle: handle,
            stream: StreamWatcher::new(handle),
        }
    }

    pub fn connect(loop_: &mut Loop, address: SocketAddr)
        -> Result<TcpWatcher, UvError>
    {
        struct Ctx { status: c_int, task: Option<BlockedTask> }

        let tcp = TcpWatcher::new(loop_);
        let ret = socket_addr_as_sockaddr(address, |addr| {
            let mut req = Request::new(uvll::UV_CONNECT);
            let result = unsafe {
                uvll::uv_tcp_connect(req.handle, tcp.handle, addr,
                                     connect_cb)
            };
            match result {
                0 => {
                    req.defuse(); // uv callback now owns this request
                    let mut cx = Ctx { status: 0, task: None };
                    wait_until_woken_after(&mut cx.task, || {
                        req.set_data(&cx);
                    });
                    match cx.status {
                        0 => Ok(()),
                        n => Err(UvError(n)),
                    }
                }
                n => Err(UvError(n))
            }
        });

        return match ret {
            Ok(()) => Ok(tcp),
            Err(e) => Err(e),
        };

        extern fn connect_cb(req: *uvll::uv_connect_t, status: c_int) {
            let req = Request::wrap(req);
            assert!(status != uvll::ECANCELED);
            let cx: &mut Ctx = unsafe { req.get_data() };
            cx.status = status;
            let scheduler: ~Scheduler = Local::take();
            scheduler.resume_blocked_task_immediately(cx.task.take_unwrap());
        }
    }
}

impl HomingIO for TcpWatcher {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl rtio::RtioSocket for TcpWatcher {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError> {
        let _m = self.fire_homing_missile();
        socket_name(Tcp, self.handle)
    }
}

impl rtio::RtioTcpStream for TcpWatcher {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        let _m = self.fire_homing_missile();
        self.stream.read(buf).map_err(uv_error_to_io_error)
    }

    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        self.stream.write(buf).map_err(uv_error_to_io_error)
    }

    fn peer_name(&mut self) -> Result<SocketAddr, IoError> {
        let _m = self.fire_homing_missile();
        socket_name(TcpPeer, self.handle)
    }

    fn control_congestion(&mut self) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            uvll::uv_tcp_nodelay(self.handle, 0 as c_int)
        })
    }

    fn nodelay(&mut self) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            uvll::uv_tcp_nodelay(self.handle, 1 as c_int)
        })
    }

    fn keepalive(&mut self, delay_in_seconds: uint) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            uvll::uv_tcp_keepalive(self.handle, 1 as c_int,
                                   delay_in_seconds as c_uint)
        })
    }

    fn letdie(&mut self) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            uvll::uv_tcp_keepalive(self.handle, 0 as c_int, 0 as c_uint)
        })
    }
}

impl UvHandle<uvll::uv_tcp_t> for TcpWatcher {
    fn uv_handle(&self) -> *uvll::uv_tcp_t { self.stream.handle }
}

impl Drop for TcpWatcher {
    fn drop(&mut self) {
        let _m = self.fire_homing_missile();
        self.close();
    }
}

// TCP listeners (unbound servers)

impl TcpListener {
    pub fn bind(loop_: &mut Loop, address: SocketAddr)
                -> Result<~TcpListener, UvError> {
        let handle = unsafe { uvll::malloc_handle(uvll::UV_TCP) };
        assert_eq!(unsafe {
            uvll::uv_tcp_init(loop_.handle, handle)
        }, 0);
        let l = ~TcpListener {
            home: get_handle_to_current_scheduler!(),
            handle: handle,
            closing_task: None,
            outgoing: Tube::new(),
        };
        let res = socket_addr_as_sockaddr(address, |addr| unsafe {
            uvll::uv_tcp_bind(l.handle, addr)
        });
        return match res {
            0 => Ok(l.install()),
            n => Err(UvError(n))
        };
    }
}

impl HomingIO for TcpListener {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl UvHandle<uvll::uv_tcp_t> for TcpListener {
    fn uv_handle(&self) -> *uvll::uv_tcp_t { self.handle }
}

impl rtio::RtioSocket for TcpListener {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError> {
        let _m = self.fire_homing_missile();
        socket_name(Tcp, self.handle)
    }
}

impl rtio::RtioTcpListener for TcpListener {
    fn listen(mut ~self) -> Result<~rtio::RtioTcpAcceptor, IoError> {
        // create the acceptor object from ourselves
        let incoming = self.outgoing.clone();
        let mut acceptor = ~TcpAcceptor {
            listener: self,
            incoming: incoming,
        };

        let _m = acceptor.fire_homing_missile();
        // XXX: the 128 backlog should be configurable
        match unsafe { uvll::uv_listen(acceptor.listener.handle, 128, listen_cb) } {
            0 => Ok(acceptor as ~rtio::RtioTcpAcceptor),
            n => Err(uv_error_to_io_error(UvError(n))),
        }
    }
}

extern fn listen_cb(server: *uvll::uv_stream_t, status: c_int) {
    assert!(status != uvll::ECANCELED);
    let msg = match status {
        0 => {
            let loop_ = Loop::wrap(unsafe {
                uvll::get_loop_for_uv_handle(server)
            });
            let client = TcpWatcher::new(&loop_);
            assert_eq!(unsafe { uvll::uv_accept(server, client.handle) }, 0);
            Ok(~client as ~rtio::RtioTcpStream)
        }
        n => Err(uv_error_to_io_error(UvError(n)))
    };

    let tcp: &mut TcpListener = unsafe { UvHandle::from_uv_handle(&server) };
    tcp.outgoing.send(msg);
}

impl Drop for TcpListener {
    fn drop(&mut self) {
        let _m = self.fire_homing_missile();
        self.close();
    }
}

extern fn listener_close_cb(handle: *uvll::uv_handle_t) {
    let tcp: &mut TcpListener = unsafe { UvHandle::from_uv_handle(&handle) };
    unsafe { uvll::free_handle(handle) }

    let sched: ~Scheduler = Local::take();
    sched.resume_blocked_task_immediately(tcp.closing_task.take_unwrap());
}

// TCP acceptors (bound servers)

impl HomingIO for TcpAcceptor {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { self.listener.home() }
}

impl rtio::RtioSocket for TcpAcceptor {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError> {
        let _m = self.fire_homing_missile();
        socket_name(Tcp, self.listener.handle)
    }
}

impl rtio::RtioTcpAcceptor for TcpAcceptor {
    fn accept(&mut self) -> Result<~rtio::RtioTcpStream, IoError> {
        let _m = self.fire_homing_missile();
        self.incoming.recv()
    }

    fn accept_simultaneously(&mut self) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            uvll::uv_tcp_simultaneous_accepts(self.listener.handle, 1)
        })
    }

    fn dont_accept_simultaneously(&mut self) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            uvll::uv_tcp_simultaneous_accepts(self.listener.handle, 0)
        })
    }
}

////////////////////////////////////////////////////////////////////////////////
/// UDP implementation
////////////////////////////////////////////////////////////////////////////////

pub struct UdpWatcher {
    handle: *uvll::uv_udp_t,
    home: SchedHandle,
}

impl UdpWatcher {
    pub fn bind(loop_: &Loop, address: SocketAddr)
                -> Result<UdpWatcher, UvError> {
        let udp = UdpWatcher {
            handle: unsafe { uvll::malloc_handle(uvll::UV_UDP) },
            home: get_handle_to_current_scheduler!(),
        };
        assert_eq!(unsafe {
            uvll::uv_udp_init(loop_.handle, udp.handle)
        }, 0);
        let result = socket_addr_as_sockaddr(address, |addr| unsafe {
            uvll::uv_udp_bind(udp.handle, addr, 0u32)
        });
        return match result {
            0 => Ok(udp),
            n => Err(UvError(n)),
        };
    }
}

impl UvHandle<uvll::uv_udp_t> for UdpWatcher {
    fn uv_handle(&self) -> *uvll::uv_udp_t { self.handle }
}

impl HomingIO for UdpWatcher {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl rtio::RtioSocket for UdpWatcher {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError> {
        let _m = self.fire_homing_missile();
        socket_name(Udp, self.handle)
    }
}

impl rtio::RtioUdpSocket for UdpWatcher {
    fn recvfrom(&mut self, buf: &mut [u8])
        -> Result<(uint, SocketAddr), IoError>
    {
        struct Ctx {
            task: Option<BlockedTask>,
            buf: Option<Buf>,
            result: Option<(ssize_t, Option<SocketAddr>)>,
        }
        let _m = self.fire_homing_missile();

        let a = match unsafe {
            uvll::uv_udp_recv_start(self.handle, alloc_cb, recv_cb)
        } {
            0 => {
                let mut cx = Ctx {
                    task: None,
                    buf: Some(slice_to_uv_buf(buf)),
                    result: None,
                };
                wait_until_woken_after(&mut cx.task, || {
                    unsafe { uvll::set_data_for_uv_handle(self.handle, &cx) }
                });
                match cx.result.take_unwrap() {
                    (n, _) if n < 0 =>
                        Err(uv_error_to_io_error(UvError(n as c_int))),
                    (n, addr) => Ok((n as uint, addr.unwrap()))
                }
            }
            n => Err(uv_error_to_io_error(UvError(n)))
        };
        return a;

        extern fn alloc_cb(handle: *uvll::uv_udp_t,
                           _suggested_size: size_t,
                           buf: *mut Buf) {
            unsafe {
                let cx: &mut Ctx =
                    cast::transmute(uvll::get_data_for_uv_handle(handle));
                *buf = cx.buf.take().expect("recv alloc_cb called more than once")
            }
        }

        extern fn recv_cb(handle: *uvll::uv_udp_t, nread: ssize_t, buf: *Buf,
                          addr: *uvll::sockaddr, _flags: c_uint) {
            assert!(nread != uvll::ECANCELED as ssize_t);
            let cx: &mut Ctx = unsafe {
                cast::transmute(uvll::get_data_for_uv_handle(handle))
            };

            // When there's no data to read the recv callback can be a no-op.
            // This can happen if read returns EAGAIN/EWOULDBLOCK. By ignoring
            // this we just drop back to kqueue and wait for the next callback.
            if nread == 0 {
                cx.buf = Some(unsafe { *buf });
                return
            }

            unsafe {
                assert_eq!(uvll::uv_udp_recv_stop(handle), 0)
            }

            let cx: &mut Ctx = unsafe {
                cast::transmute(uvll::get_data_for_uv_handle(handle))
            };
            let addr = if addr == ptr::null() {
                None
            } else {
                Some(sockaddr_to_socket_addr(addr))
            };
            cx.result = Some((nread, addr));

            let sched: ~Scheduler = Local::take();
            sched.resume_blocked_task_immediately(cx.task.take_unwrap());
        }
    }

    fn sendto(&mut self, buf: &[u8], dst: SocketAddr) -> Result<(), IoError> {
        struct Ctx { task: Option<BlockedTask>, result: c_int }

        let _m = self.fire_homing_missile();

        let mut req = Request::new(uvll::UV_UDP_SEND);
        let buf = slice_to_uv_buf(buf);
        let result = socket_addr_as_sockaddr(dst, |dst| unsafe {
            uvll::uv_udp_send(req.handle, self.handle, [buf], dst, send_cb)
        });

        return match result {
            0 => {
                req.defuse(); // uv callback now owns this request
                let mut cx = Ctx { task: None, result: 0 };
                wait_until_woken_after(&mut cx.task, || {
                    req.set_data(&cx);
                });
                match cx.result {
                    0 => Ok(()),
                    n => Err(uv_error_to_io_error(UvError(n)))
                }
            }
            n => Err(uv_error_to_io_error(UvError(n)))
        };

        extern fn send_cb(req: *uvll::uv_udp_send_t, status: c_int) {
            let req = Request::wrap(req);
            assert!(status != uvll::ECANCELED);
            let cx: &mut Ctx = unsafe { req.get_data() };
            cx.result = status;

            let sched: ~Scheduler = Local::take();
            sched.resume_blocked_task_immediately(cx.task.take_unwrap());
        }
    }

    fn join_multicast(&mut self, multi: IpAddr) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            multi.to_str().with_c_str(|m_addr| {
                uvll::uv_udp_set_membership(self.handle,
                                            m_addr, ptr::null(),
                                            uvll::UV_JOIN_GROUP)
            })
        })
    }

    fn leave_multicast(&mut self, multi: IpAddr) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            multi.to_str().with_c_str(|m_addr| {
                uvll::uv_udp_set_membership(self.handle,
                                            m_addr, ptr::null(),
                                            uvll::UV_LEAVE_GROUP)
            })
        })
    }

    fn loop_multicast_locally(&mut self) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            uvll::uv_udp_set_multicast_loop(self.handle,
                                            1 as c_int)
        })
    }

    fn dont_loop_multicast_locally(&mut self) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            uvll::uv_udp_set_multicast_loop(self.handle,
                                            0 as c_int)
        })
    }

    fn multicast_time_to_live(&mut self, ttl: int) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            uvll::uv_udp_set_multicast_ttl(self.handle,
                                           ttl as c_int)
        })
    }

    fn time_to_live(&mut self, ttl: int) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            uvll::uv_udp_set_ttl(self.handle, ttl as c_int)
        })
    }

    fn hear_broadcasts(&mut self) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            uvll::uv_udp_set_broadcast(self.handle,
                                       1 as c_int)
        })
    }

    fn ignore_broadcasts(&mut self) -> Result<(), IoError> {
        let _m = self.fire_homing_missile();
        status_to_io_result(unsafe {
            uvll::uv_udp_set_broadcast(self.handle,
                                       0 as c_int)
        })
    }
}

impl Drop for UdpWatcher {
    fn drop(&mut self) {
        // Send ourselves home to close this handle (blocking while doing so).
        let _m = self.fire_homing_missile();
        self.close();
    }
}

////////////////////////////////////////////////////////////////////////////////
/// UV request support
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use std::cell::Cell;
    use std::comm::oneshot;
    use std::rt::test::*;
    use std::rt::rtio::{RtioTcpStream, RtioTcpListener, RtioTcpAcceptor,
                        RtioUdpSocket};
    use std::task;

    use super::*;
    use super::super::local_loop;

    #[test]
    fn connect_close_ip4() {
        match TcpWatcher::connect(local_loop(), next_test_ip4()) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.name(), ~"ECONNREFUSED"),
        }
    }

    #[test]
    fn connect_close_ip6() {
        match TcpWatcher::connect(local_loop(), next_test_ip6()) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.name(), ~"ECONNREFUSED"),
        }
    }

    #[test]
    fn udp_bind_close_ip4() {
        match UdpWatcher::bind(local_loop(), next_test_ip4()) {
            Ok(..) => {}
            Err(..) => fail!()
        }
    }

    #[test]
    fn udp_bind_close_ip6() {
        match UdpWatcher::bind(local_loop(), next_test_ip6()) {
            Ok(..) => {}
            Err(..) => fail!()
        }
    }

    #[test]
    fn listen_ip4() {
        let (port, chan) = oneshot();
        let chan = Cell::new(chan);
        let addr = next_test_ip4();

        do spawn {
            let w = match TcpListener::bind(local_loop(), addr) {
                Ok(w) => w, Err(e) => fail!("{:?}", e)
            };
            let mut w = match w.listen() {
                Ok(w) => w, Err(e) => fail!("{:?}", e),
            };
            chan.take().send(());
            match w.accept() {
                Ok(mut stream) => {
                    let mut buf = [0u8, ..10];
                    match stream.read(buf) {
                        Ok(10) => {} e => fail!("{:?}", e),
                    }
                    for i in range(0, 10u8) {
                        assert_eq!(buf[i], i + 1);
                    }
                }
                Err(e) => fail!("{:?}", e)
            }
        }

        port.recv();
        let mut w = match TcpWatcher::connect(local_loop(), addr) {
            Ok(w) => w, Err(e) => fail!("{:?}", e)
        };
        match w.write([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) {
            Ok(()) => {}, Err(e) => fail!("{:?}", e)
        }
    }

    #[test]
    fn listen_ip6() {
        let (port, chan) = oneshot();
        let chan = Cell::new(chan);
        let addr = next_test_ip6();

        do spawn {
            let w = match TcpListener::bind(local_loop(), addr) {
                Ok(w) => w, Err(e) => fail!("{:?}", e)
            };
            let mut w = match w.listen() {
                Ok(w) => w, Err(e) => fail!("{:?}", e),
            };
            chan.take().send(());
            match w.accept() {
                Ok(mut stream) => {
                    let mut buf = [0u8, ..10];
                    match stream.read(buf) {
                        Ok(10) => {} e => fail!("{:?}", e),
                    }
                    for i in range(0, 10u8) {
                        assert_eq!(buf[i], i + 1);
                    }
                }
                Err(e) => fail!("{:?}", e)
            }
        }

        port.recv();
        let mut w = match TcpWatcher::connect(local_loop(), addr) {
            Ok(w) => w, Err(e) => fail!("{:?}", e)
        };
        match w.write([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) {
            Ok(()) => {}, Err(e) => fail!("{:?}", e)
        }
    }

    #[test]
    fn udp_recv_ip4() {
        let (port, chan) = oneshot();
        let chan = Cell::new(chan);
        let client = next_test_ip4();
        let server = next_test_ip4();

        do spawn {
            match UdpWatcher::bind(local_loop(), server) {
                Ok(mut w) => {
                    chan.take().send(());
                    let mut buf = [0u8, ..10];
                    match w.recvfrom(buf) {
                        Ok((10, addr)) => assert_eq!(addr, client),
                        e => fail!("{:?}", e),
                    }
                    for i in range(0, 10u8) {
                        assert_eq!(buf[i], i + 1);
                    }
                }
                Err(e) => fail!("{:?}", e)
            }
        }

        port.recv();
        let mut w = match UdpWatcher::bind(local_loop(), client) {
            Ok(w) => w, Err(e) => fail!("{:?}", e)
        };
        match w.sendto([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], server) {
            Ok(()) => {}, Err(e) => fail!("{:?}", e)
        }
    }

    #[test]
    fn udp_recv_ip6() {
        let (port, chan) = oneshot();
        let chan = Cell::new(chan);
        let client = next_test_ip6();
        let server = next_test_ip6();

        do spawn {
            match UdpWatcher::bind(local_loop(), server) {
                Ok(mut w) => {
                    chan.take().send(());
                    let mut buf = [0u8, ..10];
                    match w.recvfrom(buf) {
                        Ok((10, addr)) => assert_eq!(addr, client),
                        e => fail!("{:?}", e),
                    }
                    for i in range(0, 10u8) {
                        assert_eq!(buf[i], i + 1);
                    }
                }
                Err(e) => fail!("{:?}", e)
            }
        }

        port.recv();
        let mut w = match UdpWatcher::bind(local_loop(), client) {
            Ok(w) => w, Err(e) => fail!("{:?}", e)
        };
        match w.sendto([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], server) {
            Ok(()) => {}, Err(e) => fail!("{:?}", e)
        }
    }

    #[test]
    fn test_read_read_read() {
        use std::rt::rtio::*;
        let addr = next_test_ip4();
        static MAX: uint = 5000;
        let (port, chan) = oneshot();
        let port = Cell::new(port);
        let chan = Cell::new(chan);

        do spawn {
            let listener = TcpListener::bind(local_loop(), addr).unwrap();
            let mut acceptor = listener.listen().unwrap();
            chan.take().send(());
            let mut stream = acceptor.accept().unwrap();
            let buf = [1, .. 2048];
            let mut total_bytes_written = 0;
            while total_bytes_written < MAX {
                assert!(stream.write(buf).is_ok());
                uvdebug!("wrote bytes");
                total_bytes_written += buf.len();
            }
        }

        do spawn {
            port.take().recv();
            let mut stream = TcpWatcher::connect(local_loop(), addr).unwrap();
            let mut buf = [0, .. 2048];
            let mut total_bytes_read = 0;
            while total_bytes_read < MAX {
                let nread = stream.read(buf).unwrap();
                total_bytes_read += nread;
                for i in range(0u, nread) {
                    assert_eq!(buf[i], 1);
                }
            }
            uvdebug!("read {} bytes total", total_bytes_read);
        }
    }

    #[test]
    #[ignore(cfg(windows))] // FIXME(#10102) server never sees second packet
    fn test_udp_twice() {
        let server_addr = next_test_ip4();
        let client_addr = next_test_ip4();
        let (port, chan) = oneshot();
        let port = Cell::new(port);
        let chan = Cell::new(chan);

        do spawn {
            let mut client = UdpWatcher::bind(local_loop(), client_addr).unwrap();
            port.take().recv();
            assert!(client.sendto([1], server_addr).is_ok());
            assert!(client.sendto([2], server_addr).is_ok());
        }

        let mut server = UdpWatcher::bind(local_loop(), server_addr).unwrap();
        chan.take().send(());
        let mut buf1 = [0];
        let mut buf2 = [0];
        let (nread1, src1) = server.recvfrom(buf1).unwrap();
        let (nread2, src2) = server.recvfrom(buf2).unwrap();
        assert_eq!(nread1, 1);
        assert_eq!(nread2, 1);
        assert_eq!(src1, client_addr);
        assert_eq!(src2, client_addr);
        assert_eq!(buf1[0], 1);
        assert_eq!(buf2[0], 2);
    }

    #[test]
    fn test_udp_many_read() {
        let server_out_addr = next_test_ip4();
        let server_in_addr = next_test_ip4();
        let client_out_addr = next_test_ip4();
        let client_in_addr = next_test_ip4();
        static MAX: uint = 500_000;

        let (p1, c1) = oneshot();
        let (p2, c2) = oneshot();

        let first = Cell::new((p1, c2));
        let second = Cell::new((p2, c1));

        do spawn {
            let l = local_loop();
            let mut server_out = UdpWatcher::bind(l, server_out_addr).unwrap();
            let mut server_in = UdpWatcher::bind(l, server_in_addr).unwrap();
            let (port, chan) = first.take();
            chan.send(());
            port.recv();
            let msg = [1, .. 2048];
            let mut total_bytes_sent = 0;
            let mut buf = [1];
            while buf[0] == 1 {
                // send more data
                assert!(server_out.sendto(msg, client_in_addr).is_ok());
                total_bytes_sent += msg.len();
                // check if the client has received enough
                let res = server_in.recvfrom(buf);
                assert!(res.is_ok());
                let (nread, src) = res.unwrap();
                assert_eq!(nread, 1);
                assert_eq!(src, client_out_addr);
            }
            assert!(total_bytes_sent >= MAX);
        }

        do spawn {
            let l = local_loop();
            let mut client_out = UdpWatcher::bind(l, client_out_addr).unwrap();
            let mut client_in = UdpWatcher::bind(l, client_in_addr).unwrap();
            let (port, chan) = second.take();
            port.recv();
            chan.send(());
            let mut total_bytes_recv = 0;
            let mut buf = [0, .. 2048];
            while total_bytes_recv < MAX {
                // ask for more
                assert!(client_out.sendto([1], server_in_addr).is_ok());
                // wait for data
                let res = client_in.recvfrom(buf);
                assert!(res.is_ok());
                let (nread, src) = res.unwrap();
                assert_eq!(src, server_out_addr);
                total_bytes_recv += nread;
                for i in range(0u, nread) {
                    assert_eq!(buf[i], 1);
                }
            }
            // tell the server we're done
            assert!(client_out.sendto([0], server_in_addr).is_ok());
        }
    }

    #[test]
    fn test_read_and_block() {
        let addr = next_test_ip4();
        let (port, chan) = oneshot();
        let port = Cell::new(port);
        let chan = Cell::new(chan);

        do spawn {
            let listener = TcpListener::bind(local_loop(), addr).unwrap();
            let mut acceptor = listener.listen().unwrap();
            let (port2, chan2) = stream();
            chan.take().send(port2);
            let mut stream = acceptor.accept().unwrap();
            let mut buf = [0, .. 2048];

            let expected = 32;
            let mut current = 0;
            let mut reads = 0;

            while current < expected {
                let nread = stream.read(buf).unwrap();
                for i in range(0u, nread) {
                    let val = buf[i] as uint;
                    assert_eq!(val, current % 8);
                    current += 1;
                }
                reads += 1;

                chan2.send(());
            }

            // Make sure we had multiple reads
            assert!(reads > 1);
        }

        do spawn {
            let port2 = port.take().recv();
            let mut stream = TcpWatcher::connect(local_loop(), addr).unwrap();
            stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
            stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
            port2.recv();
            stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
            stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
            port2.recv();
        }
    }

    #[test]
    fn test_simple_tcp_server_and_client_on_diff_threads() {
        let addr = next_test_ip4();

        do task::spawn_sched(task::SingleThreaded) {
            let listener = TcpListener::bind(local_loop(), addr).unwrap();
            let mut acceptor = listener.listen().unwrap();
            let mut stream = acceptor.accept().unwrap();
            let mut buf = [0, .. 2048];
            let nread = stream.read(buf).unwrap();
            assert_eq!(nread, 8);
            for i in range(0u, nread) {
                assert_eq!(buf[i], i as u8);
            }
        }

        do task::spawn_sched(task::SingleThreaded) {
            let mut stream = TcpWatcher::connect(local_loop(), addr);
            while stream.is_err() {
                stream = TcpWatcher::connect(local_loop(), addr);
            }
            stream.unwrap().write([0, 1, 2, 3, 4, 5, 6, 7]);
        }
    }

    // On one thread, create a udp socket. Then send that socket to another
    // thread and destroy the socket on the remote thread. This should make sure
    // that homing kicks in for the socket to go back home to the original
    // thread, close itself, and then come back to the last thread.
    #[test]
    fn test_homing_closes_correctly() {
        let (port, chan) = oneshot();
        let port = Cell::new(port);
        let chan = Cell::new(chan);

        do task::spawn_sched(task::SingleThreaded) {
            let chan = Cell::new(chan.take());
            let listener = UdpWatcher::bind(local_loop(), next_test_ip4()).unwrap();
            chan.take().send(listener);
        }

        do task::spawn_sched(task::SingleThreaded) {
            let port = Cell::new(port.take());
            port.take().recv();
        }
    }

    // This is a bit of a crufty old test, but it has its uses.
    #[test]
    fn test_simple_homed_udp_io_bind_then_move_task_then_home_and_close() {
        use std::cast;
        use std::rt::local::Local;
        use std::rt::rtio::{EventLoop, IoFactory};
        use std::rt::sched::Scheduler;
        use std::rt::sched::{Shutdown, TaskFromFriend};
        use std::rt::sleeper_list::SleeperList;
        use std::rt::task::Task;
        use std::rt::task::UnwindResult;
        use std::rt::thread::Thread;
        use std::rt::deque::BufferPool;
        use std::unstable::run_in_bare_thread;
        use uvio::UvEventLoop;

        do run_in_bare_thread {
            let sleepers = SleeperList::new();
            let mut pool = BufferPool::init();
            let (worker1, stealer1) = pool.deque();
            let (worker2, stealer2) = pool.deque();
            let queues = ~[stealer1, stealer2];

            let loop1 = ~UvEventLoop::new() as ~EventLoop;
            let mut sched1 = ~Scheduler::new(loop1, worker1, queues.clone(),
                                             sleepers.clone());
            let loop2 = ~UvEventLoop::new() as ~EventLoop;
            let mut sched2 = ~Scheduler::new(loop2, worker2, queues.clone(),
                                             sleepers.clone());

            let handle1 = Cell::new(sched1.make_handle());
            let handle2 = Cell::new(sched2.make_handle());
            let tasksFriendHandle = Cell::new(sched2.make_handle());

            let on_exit: proc(UnwindResult) = proc(exit_status) {
                handle1.take().send(Shutdown);
                handle2.take().send(Shutdown);
                assert!(exit_status.is_success());
            };

            unsafe fn local_io() -> &'static mut IoFactory {
                Local::borrow(|sched: &mut Scheduler| {
                    let mut io = None;
                    sched.event_loop.io(|i| io = Some(i));
                    cast::transmute(io.unwrap())
                })
            }

            let test_function: proc() = proc() {
                let io = unsafe { local_io() };
                let addr = next_test_ip4();
                let maybe_socket = io.udp_bind(addr);
                // this socket is bound to this event loop
                assert!(maybe_socket.is_ok());

                // block self on sched1
                let scheduler: ~Scheduler = Local::take();
                scheduler.deschedule_running_task_and_then(|_, task| {
                    // unblock task
                    task.wake().map(|task| {
                        // send self to sched2
                        tasksFriendHandle.take().send(TaskFromFriend(task));
                    });
                    // sched1 should now sleep since it has nothing else to do
                })
                // sched2 will wake up and get the task as we do nothing else,
                // the function ends and the socket goes out of scope sched2
                // will start to run the destructor the destructor will first
                // block the task, set it's home as sched1, then enqueue it
                // sched2 will dequeue the task, see that it has a home, and
                // send it to sched1 sched1 will wake up, exec the close
                // function on the correct loop, and then we're done
            };

            let mut main_task = ~Task::new_root(&mut sched1.stack_pool, None,
                                                test_function);
            main_task.death.on_exit = Some(on_exit);
            let main_task = Cell::new(main_task);

            let null_task = Cell::new(~do Task::new_root(&mut sched2.stack_pool,
                                                         None) || {});

            let sched1 = Cell::new(sched1);
            let sched2 = Cell::new(sched2);

            let thread1 = do Thread::start {
                sched1.take().bootstrap(main_task.take());
            };
            let thread2 = do Thread::start {
                sched2.take().bootstrap(null_task.take());
            };

            thread1.join();
            thread2.join();
        }
    }

    #[should_fail] #[test]
    fn tcp_listener_fail_cleanup() {
        let addr = next_test_ip4();
        let w = TcpListener::bind(local_loop(), addr).unwrap();
        let _w = w.listen().unwrap();
        fail!();
    }

    #[should_fail] #[test]
    fn tcp_stream_fail_cleanup() {
        let (port, chan) = oneshot();
        let chan = Cell::new(chan);
        let addr = next_test_ip4();

        do spawn {
            let w = TcpListener::bind(local_loop(), addr).unwrap();
            let mut w = w.listen().unwrap();
            chan.take().send(());
            w.accept();
        }
        port.recv();
        let _w = TcpWatcher::connect(local_loop(), addr).unwrap();
        fail!();
    }

    #[should_fail] #[test]
    fn udp_listener_fail_cleanup() {
        let addr = next_test_ip4();
        let _w = UdpWatcher::bind(local_loop(), addr).unwrap();
        fail!();
    }

    #[should_fail] #[test]
    fn udp_fail_other_task() {
        let addr = next_test_ip4();
        let (port, chan) = oneshot();
        let chan = Cell::new(chan);

        // force the handle to be created on a different scheduler, failure in
        // the original task will force a homing operation back to this
        // scheduler.
        do task::spawn_sched(task::SingleThreaded) {
            let w = UdpWatcher::bind(local_loop(), addr).unwrap();
            chan.take().send(w);
        }

        let _w = port.recv();
        fail!();
    }

    #[should_fail]
    #[test]
    #[ignore(reason = "linked failure")]
    fn linked_failure1() {
        let (port, chan) = oneshot();
        let chan = Cell::new(chan);
        let addr = next_test_ip4();

        do spawn {
            let w = TcpListener::bind(local_loop(), addr).unwrap();
            let mut w = w.listen().unwrap();
            chan.take().send(());
            w.accept();
        }

        port.recv();
        fail!();
    }

    #[should_fail]
    #[test]
    #[ignore(reason = "linked failure")]
    fn linked_failure2() {
        let (port, chan) = oneshot();
        let chan = Cell::new(chan);
        let addr = next_test_ip4();

        do spawn {
            let w = TcpListener::bind(local_loop(), addr).unwrap();
            let mut w = w.listen().unwrap();
            chan.take().send(());
            let mut buf = [0];
            w.accept().unwrap().read(buf);
        }

        port.recv();
        let _w = TcpWatcher::connect(local_loop(), addr).unwrap();

        fail!();
    }

    #[should_fail]
    #[test]
    #[ignore(reason = "linked failure")]
    fn linked_failure3() {
        let (port, chan) = stream();
        let chan = Cell::new(chan);
        let addr = next_test_ip4();

        do spawn {
            let chan = chan.take();
            let w = TcpListener::bind(local_loop(), addr).unwrap();
            let mut w = w.listen().unwrap();
            chan.send(());
            let mut conn = w.accept().unwrap();
            chan.send(());
            let buf = [0, ..65536];
            conn.write(buf);
        }

        port.recv();
        let _w = TcpWatcher::connect(local_loop(), addr).unwrap();
        port.recv();
        fail!();
    }
}
