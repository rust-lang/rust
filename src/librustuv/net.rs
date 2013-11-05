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
use std::libc::{size_t, ssize_t, c_int, c_void, c_uint, c_char};
use std::ptr;
use std::rt::BlockedTask;
use std::rt::io::IoError;
use std::rt::io::net::ip::{Ipv4Addr, Ipv6Addr};
use std::rt::local::Local;
use std::rt::io::net::ip::{SocketAddr, IpAddr};
use std::rt::rtio;
use std::rt::sched::{Scheduler, SchedHandle};
use std::rt::tube::Tube;
use std::str;
use std::vec;

use uvll;
use uvll::*;
use super::{
            Loop, Request, UvError, Buf, NativeHandle,
            status_to_io_result,
            uv_error_to_io_error, UvHandle, slice_to_uv_buf};
use uvio::HomingIO;
use stream::StreamWatcher;

////////////////////////////////////////////////////////////////////////////////
/// Generic functions related to dealing with sockaddr things
////////////////////////////////////////////////////////////////////////////////

pub enum UvSocketAddr {
    UvIpv4SocketAddr(*sockaddr_in),
    UvIpv6SocketAddr(*sockaddr_in6),
}

pub fn sockaddr_to_UvSocketAddr(addr: *uvll::sockaddr) -> UvSocketAddr {
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
                UvIpv4SocketAddr(addr) =>
                    uvll::uv_ip4_name(addr, buf_ptr as *c_char, ip_size as size_t),
                UvIpv6SocketAddr(addr) =>
                    uvll::uv_ip6_name(addr, buf_ptr as *c_char, ip_size as size_t),
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
    use std::util;
    uv_socket_addr_as_socket_addr(addr, util::id)
}

#[cfg(test)]
#[test]
fn test_ip4_conversion() {
    use std::rt;
    let ip4 = rt::test::next_test_ip4();
    assert_eq!(ip4, socket_addr_as_uv_socket_addr(ip4, uv_socket_addr_to_socket_addr));
}

#[cfg(test)]
#[test]
fn test_ip6_conversion() {
    use std::rt;
    let ip6 = rt::test::next_test_ip6();
    assert_eq!(ip6, socket_addr_as_uv_socket_addr(ip6, uv_socket_addr_to_socket_addr));
}

enum SocketNameKind {
    TcpPeer,
    Tcp,
    Udp
}

fn socket_name(sk: SocketNameKind, handle: *c_void) -> Result<SocketAddr, IoError> {
    let getsockname = match sk {
        TcpPeer => uvll::tcp_getpeername,
        Tcp     => uvll::tcp_getsockname,
        Udp     => uvll::udp_getsockname,
    };

    // Allocate a sockaddr_storage
    // since we don't know if it's ipv4 or ipv6
    let r_addr = unsafe { uvll::malloc_sockaddr_storage() };

    let r = unsafe {
        getsockname(handle, r_addr as *uvll::sockaddr_storage)
    };

    if r != 0 {
        return Err(uv_error_to_io_error(UvError(r)));
    }

    let addr = unsafe {
        if uvll::is_ip6_addr(r_addr as *uvll::sockaddr) {
            uv_socket_addr_to_socket_addr(UvIpv6SocketAddr(r_addr as *uvll::sockaddr_in6))
        } else {
            uv_socket_addr_to_socket_addr(UvIpv4SocketAddr(r_addr as *uvll::sockaddr_in))
        }
    };

    unsafe { uvll::free_sockaddr_storage(r_addr); }

    Ok(addr)

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
            uvll::uv_tcp_init(loop_.native_handle(), handle)
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
        let ret = do socket_addr_as_uv_socket_addr(address) |addr| {
            let req = Request::new(uvll::UV_CONNECT);
            let result = match addr {
                UvIpv4SocketAddr(addr) => unsafe {
                    uvll::tcp_connect(req.handle, tcp.handle, addr,
                                      connect_cb)
                },
                UvIpv6SocketAddr(addr) => unsafe {
                    uvll::tcp_connect6(req.handle, tcp.handle, addr,
                                       connect_cb)
                },
            };
            match result {
                0 => {
                    req.defuse();
                    let mut cx = Ctx { status: 0, task: None };
                    let scheduler: ~Scheduler = Local::take();
                    do scheduler.deschedule_running_task_and_then |_, task| {
                        cx.task = Some(task);
                    }
                    match cx.status {
                        0 => Ok(()),
                        n => Err(UvError(n)),
                    }
                }
                n => Err(UvError(n))
            }
        };

        return match ret {
            Ok(()) => Ok(tcp),
            Err(e) => Err(e),
        };

        extern fn connect_cb(req: *uvll::uv_connect_t, status: c_int) {
            let _req = Request::wrap(req);
            if status == uvll::ECANCELED { return }
            let cx: &mut Ctx = unsafe {
                cast::transmute(uvll::get_data_for_req(req))
            };
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
        let _m = self.fire_missiles();
        socket_name(Tcp, self.handle)
    }
}

impl rtio::RtioTcpStream for TcpWatcher {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError> {
        let _m = self.fire_missiles();
        self.stream.read(buf).map_err(uv_error_to_io_error)
    }

    fn write(&mut self, buf: &[u8]) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        self.stream.write(buf).map_err(uv_error_to_io_error)
    }

    fn peer_name(&mut self) -> Result<SocketAddr, IoError> {
        let _m = self.fire_missiles();
        socket_name(TcpPeer, self.handle)
    }

    fn control_congestion(&mut self) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            uvll::uv_tcp_nodelay(self.handle, 0 as c_int)
        })
    }

    fn nodelay(&mut self) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            uvll::uv_tcp_nodelay(self.handle, 1 as c_int)
        })
    }

    fn keepalive(&mut self, delay_in_seconds: uint) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            uvll::uv_tcp_keepalive(self.handle, 1 as c_int,
                                   delay_in_seconds as c_uint)
        })
    }

    fn letdie(&mut self) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            uvll::uv_tcp_keepalive(self.handle, 0 as c_int, 0 as c_uint)
        })
    }
}

impl Drop for TcpWatcher {
    fn drop(&mut self) {
        let _m = self.fire_missiles();
        self.stream.close(true);
    }
}

// TCP listeners (unbound servers)

impl TcpListener {
    pub fn bind(loop_: &mut Loop, address: SocketAddr)
        -> Result<~TcpListener, UvError>
    {
        let handle = unsafe { uvll::malloc_handle(uvll::UV_TCP) };
        assert_eq!(unsafe {
            uvll::uv_tcp_init(loop_.native_handle(), handle)
        }, 0);
        let l = ~TcpListener {
            home: get_handle_to_current_scheduler!(),
            handle: handle,
            closing_task: None,
            outgoing: Tube::new(),
        };
        let res = socket_addr_as_uv_socket_addr(address, |addr| unsafe {
            match addr {
                UvIpv4SocketAddr(addr) => uvll::tcp_bind(l.handle, addr),
                UvIpv6SocketAddr(addr) => uvll::tcp_bind6(l.handle, addr),
            }
        });
        match res {
            0 => Ok(l.install()),
            n => Err(UvError(n))
        }
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
        let _m = self.fire_missiles();
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

        let _m = acceptor.fire_missiles();
        // XXX: the 128 backlog should be configurable
        match unsafe { uvll::uv_listen(acceptor.listener.handle, 128, listen_cb) } {
            0 => Ok(acceptor as ~rtio::RtioTcpAcceptor),
            n => Err(uv_error_to_io_error(UvError(n))),
        }
    }
}

extern fn listen_cb(server: *uvll::uv_stream_t, status: c_int) {
    let msg = match status {
        0 => {
            let loop_ = NativeHandle::from_native_handle(unsafe {
                uvll::get_loop_for_uv_handle(server)
            });
            let client = TcpWatcher::new(&loop_);
            assert_eq!(unsafe { uvll::uv_accept(server, client.handle) }, 0);
            Ok(~client as ~rtio::RtioTcpStream)
        }
        uvll::ECANCELED => return,
        n => Err(uv_error_to_io_error(UvError(n)))
    };

    let tcp: &mut TcpListener = unsafe { UvHandle::from_uv_handle(&server) };
    tcp.outgoing.send(msg);
}

impl Drop for TcpListener {
    fn drop(&mut self) {
        let (_m, sched) = self.fire_missiles_sched();

        do sched.deschedule_running_task_and_then |_, task| {
            self.closing_task = Some(task);
            unsafe { uvll::uv_close(self.handle, listener_close_cb) }
        }
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
        let _m = self.fire_missiles();
        socket_name(Tcp, self.listener.handle)
    }
}

impl rtio::RtioTcpAcceptor for TcpAcceptor {
    fn accept(&mut self) -> Result<~rtio::RtioTcpStream, IoError> {
        let _m = self.fire_missiles();
        self.incoming.recv()
    }

    fn accept_simultaneously(&mut self) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            uvll::uv_tcp_simultaneous_accepts(self.listener.handle, 1)
        })
    }

    fn dont_accept_simultaneously(&mut self) -> Result<(), IoError> {
        let _m = self.fire_missiles();
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
        -> Result<UdpWatcher, UvError>
    {
        let udp = UdpWatcher {
            handle: unsafe { uvll::malloc_handle(uvll::UV_UDP) },
            home: get_handle_to_current_scheduler!(),
        };
        assert_eq!(unsafe {
            uvll::uv_udp_init(loop_.native_handle(), udp.handle)
        }, 0);
        let result = socket_addr_as_uv_socket_addr(address, |addr| unsafe {
            match addr {
                UvIpv4SocketAddr(addr) => uvll::udp_bind(udp.handle, addr, 0u32),
                UvIpv6SocketAddr(addr) => uvll::udp_bind6(udp.handle, addr, 0u32),
            }
        });
        match result {
            0 => Ok(udp),
            n => Err(UvError(n)),
        }
    }
}

impl HomingIO for UdpWatcher {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl rtio::RtioSocket for UdpWatcher {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError> {
        let _m = self.fire_missiles();
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
            result: Option<(ssize_t, SocketAddr)>,
        }
        let _m = self.fire_missiles();

        return match unsafe {
            uvll::uv_udp_recv_start(self.handle, alloc_cb, recv_cb)
        } {
            0 => {
                let mut cx = Ctx {
                    task: None,
                    buf: Some(slice_to_uv_buf(buf)),
                    result: None,
                };
                unsafe { uvll::set_data_for_uv_handle(self.handle, &cx) }
                let scheduler: ~Scheduler = Local::take();
                do scheduler.deschedule_running_task_and_then |_, task| {
                    cx.task = Some(task);
                }
                match cx.result.take_unwrap() {
                    (n, _) if n < 0 =>
                        Err(uv_error_to_io_error(UvError(n as c_int))),
                    (n, addr) => Ok((n as uint, addr))
                }
            }
            n => Err(uv_error_to_io_error(UvError(n)))
        };

        extern fn alloc_cb(handle: *uvll::uv_udp_t,
                           _suggested_size: size_t) -> Buf {
            let cx: &mut Ctx = unsafe {
                cast::transmute(uvll::get_data_for_uv_handle(handle))
            };
            cx.buf.take().expect("alloc_cb called more than once")
        }

        extern fn recv_cb(handle: *uvll::uv_udp_t, nread: ssize_t, _buf: Buf,
                          addr: *uvll::sockaddr, _flags: c_uint) {

            // When there's no data to read the recv callback can be a no-op.
            // This can happen if read returns EAGAIN/EWOULDBLOCK. By ignoring
            // this we just drop back to kqueue and wait for the next callback.
            if nread == 0 { return }
            if nread == uvll::ECANCELED as ssize_t { return }

            unsafe {
                assert_eq!(uvll::uv_udp_recv_stop(handle), 0)
            }

            let cx: &mut Ctx = unsafe {
                cast::transmute(uvll::get_data_for_uv_handle(handle))
            };
            let addr = sockaddr_to_UvSocketAddr(addr);
            let addr = uv_socket_addr_to_socket_addr(addr);
            cx.result = Some((nread, addr));

            let sched: ~Scheduler = Local::take();
            sched.resume_blocked_task_immediately(cx.task.take_unwrap());
        }
    }

    fn sendto(&mut self, buf: &[u8], dst: SocketAddr) -> Result<(), IoError> {
        struct Ctx { task: Option<BlockedTask>, result: c_int }

        let _m = self.fire_missiles();

        let req = Request::new(uvll::UV_UDP_SEND);
        let buf = slice_to_uv_buf(buf);
        let result = socket_addr_as_uv_socket_addr(dst, |dst| unsafe {
            match dst {
                UvIpv4SocketAddr(dst) =>
                    uvll::udp_send(req.handle, self.handle, [buf], dst, send_cb),
                UvIpv6SocketAddr(dst) =>
                    uvll::udp_send6(req.handle, self.handle, [buf], dst, send_cb),
            }
        });

        return match result {
            0 => {
                let mut cx = Ctx { task: None, result: 0 };
                req.set_data(&cx);
                req.defuse();

                let sched: ~Scheduler = Local::take();
                do sched.deschedule_running_task_and_then |_, task| {
                    cx.task = Some(task);
                }

                match cx.result {
                    0 => Ok(()),
                    n => Err(uv_error_to_io_error(UvError(n)))
                }
            }
            n => Err(uv_error_to_io_error(UvError(n)))
        };

        extern fn send_cb(req: *uvll::uv_udp_send_t, status: c_int) {
            let req = Request::wrap(req);
            let cx: &mut Ctx = unsafe { cast::transmute(req.get_data()) };
            cx.result = status;

            let sched: ~Scheduler = Local::take();
            sched.resume_blocked_task_immediately(cx.task.take_unwrap());
        }
    }

    fn join_multicast(&mut self, multi: IpAddr) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            do multi.to_str().with_c_str |m_addr| {
                uvll::uv_udp_set_membership(self.handle,
                                            m_addr, ptr::null(),
                                            uvll::UV_JOIN_GROUP)
            }
        })
    }

    fn leave_multicast(&mut self, multi: IpAddr) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            do multi.to_str().with_c_str |m_addr| {
                uvll::uv_udp_set_membership(self.handle,
                                            m_addr, ptr::null(),
                                            uvll::UV_LEAVE_GROUP)
            }
        })
    }

    fn loop_multicast_locally(&mut self) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            uvll::uv_udp_set_multicast_loop(self.handle,
                                            1 as c_int)
        })
    }

    fn dont_loop_multicast_locally(&mut self) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            uvll::uv_udp_set_multicast_loop(self.handle,
                                            0 as c_int)
        })
    }

    fn multicast_time_to_live(&mut self, ttl: int) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            uvll::uv_udp_set_multicast_ttl(self.handle,
                                           ttl as c_int)
        })
    }

    fn time_to_live(&mut self, ttl: int) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            uvll::uv_udp_set_ttl(self.handle, ttl as c_int)
        })
    }

    fn hear_broadcasts(&mut self) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            uvll::uv_udp_set_broadcast(self.handle,
                                       1 as c_int)
        })
    }

    fn ignore_broadcasts(&mut self) -> Result<(), IoError> {
        let _m = self.fire_missiles();
        status_to_io_result(unsafe {
            uvll::uv_udp_set_broadcast(self.handle,
                                       0 as c_int)
        })
    }
}

impl Drop for UdpWatcher {
    fn drop(&mut self) {
        // Send ourselves home to close this handle (blocking while doing so).
        let (_m, sched) = self.fire_missiles_sched();
        let mut slot = None;
        unsafe {
            uvll::set_data_for_uv_handle(self.handle, &slot);
            uvll::uv_close(self.handle, close_cb);
        }
        do sched.deschedule_running_task_and_then |_, task| {
            slot = Some(task);
        }

        extern fn close_cb(handle: *uvll::uv_handle_t) {
            let slot: &mut Option<BlockedTask> = unsafe {
                cast::transmute(uvll::get_data_for_uv_handle(handle))
            };
            let sched: ~Scheduler = Local::take();
            sched.resume_blocked_task_immediately(slot.take_unwrap());
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// UV request support
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use std::util::ignore;
    use std::cell::Cell;
    use std::vec;
    use std::unstable::run_in_bare_thread;
    use std::rt::thread::Thread;
    use std::rt::test::*;
    use super::super::{Loop, AllocCallback};
    use super::super::{vec_from_uv_buf, vec_to_uv_buf, slice_to_uv_buf};

    #[test]
    fn connect_close_ip4() {
        do run_in_bare_thread() {
            let mut loop_ = Loop::new();
            let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
            // Connect to a port where nobody is listening
            let addr = next_test_ip4();
            do tcp_watcher.connect(addr) |stream_watcher, status| {
                uvdebug!("tcp_watcher.connect!");
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
                uvdebug!("tcp_watcher.connect!");
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
            uvdebug!("listening");
            let mut stream = server_tcp_watcher.as_stream();
            let res = do stream.listen |mut server_stream_watcher, status| {
                uvdebug!("listened!");
                assert!(status.is_none());
                let mut loop_ = loop_;
                let client_tcp_watcher = TcpWatcher::new(&mut loop_);
                let mut client_tcp_watcher = client_tcp_watcher.as_stream();
                server_stream_watcher.accept(client_tcp_watcher);
                let count_cell = Cell::new(0);
                let server_stream_watcher = server_stream_watcher;
                uvdebug!("starting read");
                let alloc: AllocCallback = |size| {
                    vec_to_uv_buf(vec::from_elem(size, 0u8))
                };
                do client_tcp_watcher.read_start(alloc) |stream_watcher, nread, buf, status| {

                    uvdebug!("i'm reading!");
                    let buf = vec_from_uv_buf(buf);
                    let mut count = count_cell.take();
                    if status.is_none() {
                        uvdebug!("got {} bytes", nread);
                        let buf = buf.unwrap();
                        for byte in buf.slice(0, nread as uint).iter() {
                            assert!(*byte == count as u8);
                            uvdebug!("{}", *byte as uint);
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
            };

            assert!(res.is_ok());

            let client_thread = do Thread::start {
                uvdebug!("starting client thread");
                let mut loop_ = Loop::new();
                let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
                do tcp_watcher.connect(addr) |mut stream_watcher, status| {
                    uvdebug!("connecting");
                    assert!(status.is_none());
                    let msg = ~[0, 1, 2, 3, 4, 5, 6 ,7 ,8, 9];
                    let buf = slice_to_uv_buf(msg);
                    let msg_cell = Cell::new(msg);
                    do stream_watcher.write(buf) |stream_watcher, status| {
                        uvdebug!("writing");
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
        };
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
            uvdebug!("listening");
            let mut stream = server_tcp_watcher.as_stream();
            let res = do stream.listen |mut server_stream_watcher, status| {
                uvdebug!("listened!");
                assert!(status.is_none());
                let mut loop_ = loop_;
                let client_tcp_watcher = TcpWatcher::new(&mut loop_);
                let mut client_tcp_watcher = client_tcp_watcher.as_stream();
                server_stream_watcher.accept(client_tcp_watcher);
                let count_cell = Cell::new(0);
                let server_stream_watcher = server_stream_watcher;
                uvdebug!("starting read");
                let alloc: AllocCallback = |size| {
                    vec_to_uv_buf(vec::from_elem(size, 0u8))
                };
                do client_tcp_watcher.read_start(alloc)
                    |stream_watcher, nread, buf, status| {

                    uvdebug!("i'm reading!");
                    let buf = vec_from_uv_buf(buf);
                    let mut count = count_cell.take();
                    if status.is_none() {
                        uvdebug!("got {} bytes", nread);
                        let buf = buf.unwrap();
                        let r = buf.slice(0, nread as uint);
                        for byte in r.iter() {
                            assert!(*byte == count as u8);
                            uvdebug!("{}", *byte as uint);
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
            };
            assert!(res.is_ok());

            let client_thread = do Thread::start {
                uvdebug!("starting client thread");
                let mut loop_ = Loop::new();
                let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
                do tcp_watcher.connect(addr) |mut stream_watcher, status| {
                    uvdebug!("connecting");
                    assert!(status.is_none());
                    let msg = ~[0, 1, 2, 3, 4, 5, 6 ,7 ,8, 9];
                    let buf = slice_to_uv_buf(msg);
                    let msg_cell = Cell::new(msg);
                    do stream_watcher.write(buf) |stream_watcher, status| {
                        uvdebug!("writing");
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

            uvdebug!("starting read");
            let alloc: AllocCallback = |size| {
                vec_to_uv_buf(vec::from_elem(size, 0u8))
            };

            do server.recv_start(alloc) |mut server, nread, buf, src, flags, status| {
                server.recv_stop();
                uvdebug!("i'm reading!");
                assert!(status.is_none());
                assert_eq!(flags, 0);
                assert_eq!(src, client_addr);

                let buf = vec_from_uv_buf(buf);
                let mut count = 0;
                uvdebug!("got {} bytes", nread);

                let buf = buf.unwrap();
                for &byte in buf.slice(0, nread as uint).iter() {
                    assert!(byte == count as u8);
                    uvdebug!("{}", byte as uint);
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
                    uvdebug!("writing");
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

            uvdebug!("starting read");
            let alloc: AllocCallback = |size| {
                vec_to_uv_buf(vec::from_elem(size, 0u8))
            };

            do server.recv_start(alloc) |mut server, nread, buf, src, flags, status| {
                server.recv_stop();
                uvdebug!("i'm reading!");
                assert!(status.is_none());
                assert_eq!(flags, 0);
                assert_eq!(src, client_addr);

                let buf = vec_from_uv_buf(buf);
                let mut count = 0;
                uvdebug!("got {} bytes", nread);

                let buf = buf.unwrap();
                for &byte in buf.slice(0, nread as uint).iter() {
                    assert!(byte == count as u8);
                    uvdebug!("{}", byte as uint);
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
                    uvdebug!("writing");
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
