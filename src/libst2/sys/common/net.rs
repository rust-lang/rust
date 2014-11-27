// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::SocketStatus::*;
pub use self::InAddr::*;

use alloc::arc::Arc;
use libc::{mod, c_char, c_int};
use mem;
use num::Int;
use ptr::{mod, null, null_mut};
use rustrt::mutex;
use io::net::ip::{SocketAddr, IpAddr, Ipv4Addr, Ipv6Addr};
use io::net::addrinfo;
use io::{IoResult, IoError};
use sys::{mod, retry, c, sock_t, last_error, last_net_error, last_gai_error, close_sock,
          wrlen, msglen_t, os, wouldblock, set_nonblocking, timer, ms_to_timeval,
          decode_error_detailed};
use sys_common::{mod, keep_going, short_write, timeout};
use prelude::*;
use cmp;
use io;

// FIXME: move uses of Arc and deadline tracking to std::io

#[deriving(Show)]
pub enum SocketStatus {
    Readable,
    Writable,
}

////////////////////////////////////////////////////////////////////////////////
// sockaddr and misc bindings
////////////////////////////////////////////////////////////////////////////////

pub fn htons(u: u16) -> u16 { unimplemented!() }
pub fn ntohs(u: u16) -> u16 { unimplemented!() }

pub enum InAddr {
    In4Addr(libc::in_addr),
    In6Addr(libc::in6_addr),
}

pub fn ip_to_inaddr(ip: IpAddr) -> InAddr { unimplemented!() }

pub fn addr_to_sockaddr(addr: SocketAddr,
                    storage: &mut libc::sockaddr_storage)
                    -> libc::socklen_t { unimplemented!() }

pub fn socket(addr: SocketAddr, ty: libc::c_int) -> IoResult<sock_t> { unimplemented!() }

pub fn setsockopt<T>(fd: sock_t, opt: libc::c_int, val: libc::c_int,
                 payload: T) -> IoResult<()> { unimplemented!() }

pub fn getsockopt<T: Copy>(fd: sock_t, opt: libc::c_int,
                           val: libc::c_int) -> IoResult<T> { unimplemented!() }

pub fn sockname(fd: sock_t,
            f: unsafe extern "system" fn(sock_t, *mut libc::sockaddr,
                                         *mut libc::socklen_t) -> libc::c_int)
    -> IoResult<SocketAddr>
{ unimplemented!() }

pub fn sockaddr_to_addr(storage: &libc::sockaddr_storage,
                        len: uint) -> IoResult<SocketAddr> { unimplemented!() }

////////////////////////////////////////////////////////////////////////////////
// get_host_addresses
////////////////////////////////////////////////////////////////////////////////

extern "system" {
    fn getaddrinfo(node: *const c_char, service: *const c_char,
                   hints: *const libc::addrinfo,
                   res: *mut *mut libc::addrinfo) -> c_int;
    fn freeaddrinfo(res: *mut libc::addrinfo);
}

pub fn get_host_addresses(host: Option<&str>, servname: Option<&str>,
                          hint: Option<addrinfo::Hint>)
                          -> Result<Vec<addrinfo::Info>, IoError>
{ unimplemented!() }

////////////////////////////////////////////////////////////////////////////////
// Timeout helpers
//
// The read/write functions below are the helpers for reading/writing a socket
// with a possible deadline specified. This is generally viewed as a timed out
// I/O operation.
//
// From the application's perspective, timeouts apply to the I/O object, not to
// the underlying file descriptor (it's one timeout per object). This means that
// we can't use the SO_RCVTIMEO and corresponding send timeout option.
//
// The next idea to implement timeouts would be to use nonblocking I/O. An
// invocation of select() would wait (with a timeout) for a socket to be ready.
// Once its ready, we can perform the operation. Note that the operation *must*
// be nonblocking, even though select() says the socket is ready. This is
// because some other thread could have come and stolen our data (handles can be
// cloned).
//
// To implement nonblocking I/O, the first option we have is to use the
// O_NONBLOCK flag. Remember though that this is a global setting, affecting all
// I/O objects, so this was initially viewed as unwise.
//
// It turns out that there's this nifty MSG_DONTWAIT flag which can be passed to
// send/recv, but the niftiness wears off once you realize it only works well on
// Linux [1] [2]. This means that it's pretty easy to get a nonblocking
// operation on Linux (no flag fiddling, no affecting other objects), but not on
// other platforms.
//
// To work around this constraint on other platforms, we end up using the
// original strategy of flipping the O_NONBLOCK flag. As mentioned before, this
// could cause other objects' blocking operations to suddenly become
// nonblocking. To get around this, a "blocking operation" which returns EAGAIN
// falls back to using the same code path as nonblocking operations, but with an
// infinite timeout (select + send/recv). This helps emulate blocking
// reads/writes despite the underlying descriptor being nonblocking, as well as
// optimizing the fast path of just hitting one syscall in the good case.
//
// As a final caveat, this implementation uses a mutex so only one thread is
// doing a nonblocking operation at at time. This is the operation that comes
// after the select() (at which point we think the socket is ready). This is
// done for sanity to ensure that the state of the O_NONBLOCK flag is what we
// expect (wouldn't want someone turning it on when it should be off!). All
// operations performed in the lock are *nonblocking* to avoid holding the mutex
// forever.
//
// So, in summary, Linux uses MSG_DONTWAIT and doesn't need mutexes, everyone
// else uses O_NONBLOCK and mutexes with some trickery to make sure blocking
// reads/writes are still blocking.
//
// Fun, fun!
//
// [1] http://twistedmatrix.com/pipermail/twisted-commits/2012-April/034692.html
// [2] http://stackoverflow.com/questions/19819198/does-send-msg-dontwait

pub fn read<T>(fd: sock_t,
               deadline: u64,
               lock: || -> T,
               read: |bool| -> libc::c_int) -> IoResult<uint> { unimplemented!() }

pub fn write<T>(fd: sock_t,
                deadline: u64,
                buf: &[u8],
                write_everything: bool,
                lock: || -> T,
                write: |bool, *const u8, uint| -> i64) -> IoResult<uint> { unimplemented!() }

// See http://developerweb.net/viewtopic.php?id=3196 for where this is
// derived from.
pub fn connect_timeout(fd: sock_t,
                       addrp: *const libc::sockaddr,
                       len: libc::socklen_t,
                       timeout_ms: u64) -> IoResult<()> { unimplemented!() }

pub fn await(fds: &[sock_t], deadline: Option<u64>,
             status: SocketStatus) -> IoResult<()> { unimplemented!() }

////////////////////////////////////////////////////////////////////////////////
// Basic socket representation
////////////////////////////////////////////////////////////////////////////////

struct Inner {
    fd: sock_t,

    // Unused on Linux, where this lock is not necessary.
    #[allow(dead_code)]
    lock: mutex::NativeMutex
}

impl Inner {
    fn new(fd: sock_t) -> Inner { unimplemented!() }
}

impl Drop for Inner {
    fn drop(&mut self) { unimplemented!() }
}

pub struct Guard<'a> {
    pub fd: sock_t,
    pub guard: mutex::LockGuard<'a>,
}

#[unsafe_destructor]
impl<'a> Drop for Guard<'a> {
    fn drop(&mut self) { unimplemented!() }
}

////////////////////////////////////////////////////////////////////////////////
// TCP streams
////////////////////////////////////////////////////////////////////////////////

pub struct TcpStream {
    inner: Arc<Inner>,
    read_deadline: u64,
    write_deadline: u64,
}

impl TcpStream {
    pub fn connect(addr: SocketAddr, timeout: Option<u64>) -> IoResult<TcpStream> { unimplemented!() }

    pub fn new(fd: sock_t) -> TcpStream { unimplemented!() }

    pub fn fd(&self) -> sock_t { unimplemented!() }

    pub fn set_nodelay(&mut self, nodelay: bool) -> IoResult<()> { unimplemented!() }

    pub fn set_keepalive(&mut self, seconds: Option<uint>) -> IoResult<()> { unimplemented!() }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn set_tcp_keepalive(&mut self, seconds: uint) -> IoResult<()> { unimplemented!() }
    #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
    fn set_tcp_keepalive(&mut self, seconds: uint) -> IoResult<()> { unimplemented!() }
    #[cfg(not(any(target_os = "macos",
                  target_os = "ios",
                  target_os = "freebsd",
                  target_os = "dragonfly")))]
    fn set_tcp_keepalive(&mut self, _seconds: uint) -> IoResult<()> { unimplemented!() }

    #[cfg(target_os = "linux")]
    fn lock_nonblocking(&self) { unimplemented!() }

    #[cfg(not(target_os = "linux"))]
    fn lock_nonblocking<'a>(&'a self) -> Guard<'a> { unimplemented!() }

    pub fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }

    pub fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }
    pub fn peer_name(&mut self) -> IoResult<SocketAddr> { unimplemented!() }

    pub fn close_write(&mut self) -> IoResult<()> { unimplemented!() }
    pub fn close_read(&mut self) -> IoResult<()> { unimplemented!() }

    pub fn set_timeout(&mut self, timeout: Option<u64>) { unimplemented!() }
    pub fn set_read_timeout(&mut self, timeout: Option<u64>) { unimplemented!() }
    pub fn set_write_timeout(&mut self, timeout: Option<u64>) { unimplemented!() }

    pub fn socket_name(&mut self) -> IoResult<SocketAddr> { unimplemented!() }
}

impl Clone for TcpStream {
    fn clone(&self) -> TcpStream { unimplemented!() }
}

////////////////////////////////////////////////////////////////////////////////
// UDP
////////////////////////////////////////////////////////////////////////////////

pub struct UdpSocket {
    inner: Arc<Inner>,
    read_deadline: u64,
    write_deadline: u64,
}

impl UdpSocket {
    pub fn bind(addr: SocketAddr) -> IoResult<UdpSocket> { unimplemented!() }

    pub fn fd(&self) -> sock_t { unimplemented!() }

    pub fn set_broadcast(&mut self, on: bool) -> IoResult<()> { unimplemented!() }

    pub fn set_multicast_loop(&mut self, on: bool) -> IoResult<()> { unimplemented!() }

    pub fn set_membership(&mut self, addr: IpAddr, opt: libc::c_int) -> IoResult<()> { unimplemented!() }

    #[cfg(target_os = "linux")]
    fn lock_nonblocking(&self) { unimplemented!() }

    #[cfg(not(target_os = "linux"))]
    fn lock_nonblocking<'a>(&'a self) -> Guard<'a> { unimplemented!() }

    pub fn socket_name(&mut self) -> IoResult<SocketAddr> { unimplemented!() }

    pub fn recv_from(&mut self, buf: &mut [u8]) -> IoResult<(uint, SocketAddr)> { unimplemented!() }

    pub fn send_to(&mut self, buf: &[u8], dst: SocketAddr) -> IoResult<()> { unimplemented!() }

    pub fn join_multicast(&mut self, multi: IpAddr) -> IoResult<()> { unimplemented!() }
    pub fn leave_multicast(&mut self, multi: IpAddr) -> IoResult<()> { unimplemented!() }

    pub fn multicast_time_to_live(&mut self, ttl: int) -> IoResult<()> { unimplemented!() }
    pub fn time_to_live(&mut self, ttl: int) -> IoResult<()> { unimplemented!() }

    pub fn set_timeout(&mut self, timeout: Option<u64>) { unimplemented!() }
    pub fn set_read_timeout(&mut self, timeout: Option<u64>) { unimplemented!() }
    pub fn set_write_timeout(&mut self, timeout: Option<u64>) { unimplemented!() }
}

impl Clone for UdpSocket {
    fn clone(&self) -> UdpSocket { unimplemented!() }
}
