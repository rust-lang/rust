// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;
use self::SocketStatus::*;
use self::InAddr::*;

use ffi::{CString, CStr};
use old_io::net::addrinfo;
use old_io::net::ip::{SocketAddr, IpAddr, Ipv4Addr, Ipv6Addr};
use old_io::{IoResult, IoError};
use libc::{self, c_char, c_int};
use mem;
use num::Int;
use ptr::{self, null, null_mut};
use str;
use sys::{self, retry, c, sock_t, last_error, last_net_error, last_gai_error, close_sock,
          wrlen, msglen_t, os, wouldblock, set_nonblocking, timer, ms_to_timeval,
          decode_error_detailed};
use sync::{Arc, Mutex, MutexGuard};
use sys_common::{self, keep_going, short_write, timeout};
use cmp;
use old_io;

// FIXME: move uses of Arc and deadline tracking to std::io

#[derive(Debug)]
pub enum SocketStatus {
    Readable,
    Writable,
}

////////////////////////////////////////////////////////////////////////////////
// sockaddr and misc bindings
////////////////////////////////////////////////////////////////////////////////

pub fn htons(u: u16) -> u16 {
    u.to_be()
}
pub fn ntohs(u: u16) -> u16 {
    Int::from_be(u)
}

pub enum InAddr {
    In4Addr(libc::in_addr),
    In6Addr(libc::in6_addr),
}

pub fn ip_to_inaddr(ip: IpAddr) -> InAddr {
    match ip {
        Ipv4Addr(a, b, c, d) => {
            let ip = ((a as u32) << 24) |
                     ((b as u32) << 16) |
                     ((c as u32) <<  8) |
                     ((d as u32) <<  0);
            In4Addr(libc::in_addr {
                s_addr: Int::from_be(ip)
            })
        }
        Ipv6Addr(a, b, c, d, e, f, g, h) => {
            In6Addr(libc::in6_addr {
                s6_addr: [
                    htons(a),
                    htons(b),
                    htons(c),
                    htons(d),
                    htons(e),
                    htons(f),
                    htons(g),
                    htons(h),
                ]
            })
        }
    }
}

pub fn addr_to_sockaddr(addr: SocketAddr,
                    storage: &mut libc::sockaddr_storage)
                    -> libc::socklen_t {
    unsafe {
        let len = match ip_to_inaddr(addr.ip) {
            In4Addr(inaddr) => {
                let storage = storage as *mut _ as *mut libc::sockaddr_in;
                (*storage).sin_family = libc::AF_INET as libc::sa_family_t;
                (*storage).sin_port = htons(addr.port);
                (*storage).sin_addr = inaddr;
                mem::size_of::<libc::sockaddr_in>()
            }
            In6Addr(inaddr) => {
                let storage = storage as *mut _ as *mut libc::sockaddr_in6;
                (*storage).sin6_family = libc::AF_INET6 as libc::sa_family_t;
                (*storage).sin6_port = htons(addr.port);
                (*storage).sin6_addr = inaddr;
                mem::size_of::<libc::sockaddr_in6>()
            }
        };
        return len as libc::socklen_t;
    }
}

pub fn socket(addr: SocketAddr, ty: libc::c_int) -> IoResult<sock_t> {
    unsafe {
        let fam = match addr.ip {
            Ipv4Addr(..) => libc::AF_INET,
            Ipv6Addr(..) => libc::AF_INET6,
        };
        match libc::socket(fam, ty, 0) {
            -1 => Err(last_net_error()),
            fd => Ok(fd),
        }
    }
}

pub fn setsockopt<T>(fd: sock_t, opt: libc::c_int, val: libc::c_int,
                 payload: T) -> IoResult<()> {
    unsafe {
        let payload = &payload as *const T as *const libc::c_void;
        let ret = libc::setsockopt(fd, opt, val,
                                   payload,
                                   mem::size_of::<T>() as libc::socklen_t);
        if ret != 0 {
            Err(last_net_error())
        } else {
            Ok(())
        }
    }
}

pub fn getsockopt<T: Copy>(fd: sock_t, opt: libc::c_int,
                           val: libc::c_int) -> IoResult<T> {
    unsafe {
        let mut slot: T = mem::zeroed();
        let mut len = mem::size_of::<T>() as libc::socklen_t;
        let ret = c::getsockopt(fd, opt, val,
                                &mut slot as *mut _ as *mut _,
                                &mut len);
        if ret != 0 {
            Err(last_net_error())
        } else {
            assert!(len as uint == mem::size_of::<T>());
            Ok(slot)
        }
    }
}

pub fn sockname(fd: sock_t,
            f: unsafe extern "system" fn(sock_t, *mut libc::sockaddr,
                                         *mut libc::socklen_t) -> libc::c_int)
    -> IoResult<SocketAddr>
{
    let mut storage: libc::sockaddr_storage = unsafe { mem::zeroed() };
    let mut len = mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    unsafe {
        let storage = &mut storage as *mut libc::sockaddr_storage;
        let ret = f(fd,
                    storage as *mut libc::sockaddr,
                    &mut len as *mut libc::socklen_t);
        if ret != 0 {
            return Err(last_net_error())
        }
    }
    return sockaddr_to_addr(&storage, len as uint);
}

pub fn sockaddr_to_addr(storage: &libc::sockaddr_storage,
                        len: uint) -> IoResult<SocketAddr> {
    match storage.ss_family as libc::c_int {
        libc::AF_INET => {
            assert!(len as uint >= mem::size_of::<libc::sockaddr_in>());
            let storage: &libc::sockaddr_in = unsafe {
                mem::transmute(storage)
            };
            let ip = (storage.sin_addr.s_addr as u32).to_be();
            let a = (ip >> 24) as u8;
            let b = (ip >> 16) as u8;
            let c = (ip >>  8) as u8;
            let d = (ip >>  0) as u8;
            Ok(SocketAddr {
                ip: Ipv4Addr(a, b, c, d),
                port: ntohs(storage.sin_port),
            })
        }
        libc::AF_INET6 => {
            assert!(len as uint >= mem::size_of::<libc::sockaddr_in6>());
            let storage: &libc::sockaddr_in6 = unsafe {
                mem::transmute(storage)
            };
            let a = ntohs(storage.sin6_addr.s6_addr[0]);
            let b = ntohs(storage.sin6_addr.s6_addr[1]);
            let c = ntohs(storage.sin6_addr.s6_addr[2]);
            let d = ntohs(storage.sin6_addr.s6_addr[3]);
            let e = ntohs(storage.sin6_addr.s6_addr[4]);
            let f = ntohs(storage.sin6_addr.s6_addr[5]);
            let g = ntohs(storage.sin6_addr.s6_addr[6]);
            let h = ntohs(storage.sin6_addr.s6_addr[7]);
            Ok(SocketAddr {
                ip: Ipv6Addr(a, b, c, d, e, f, g, h),
                port: ntohs(storage.sin6_port),
            })
        }
        _ => {
            Err(IoError {
                kind: old_io::InvalidInput,
                desc: "invalid argument",
                detail: None,
            })
        }
    }
}

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
{
    sys::init_net();

    assert!(host.is_some() || servname.is_some());

    let c_host = match host {
        Some(x) => Some(try!(CString::new(x))),
        None => None,
    };
    let c_host = c_host.as_ref().map(|x| x.as_ptr()).unwrap_or(null());
    let c_serv = match servname {
        Some(x) => Some(try!(CString::new(x))),
        None => None,
    };
    let c_serv = c_serv.as_ref().map(|x| x.as_ptr()).unwrap_or(null());

    let hint = hint.map(|hint| {
        libc::addrinfo {
            ai_flags: hint.flags as c_int,
            ai_family: hint.family as c_int,
            ai_socktype: 0,
            ai_protocol: 0,
            ai_addrlen: 0,
            ai_canonname: null_mut(),
            ai_addr: null_mut(),
            ai_next: null_mut()
        }
    });

    let hint_ptr = hint.as_ref().map_or(null(), |x| {
        x as *const libc::addrinfo
    });
    let mut res = null_mut();

    // Make the call
    let s = unsafe {
        getaddrinfo(c_host, c_serv, hint_ptr, &mut res)
    };

    // Error?
    if s != 0 {
        return Err(last_gai_error(s));
    }

    // Collect all the results we found
    let mut addrs = Vec::new();
    let mut rp = res;
    while !rp.is_null() {
        unsafe {
            let addr = try!(sockaddr_to_addr(mem::transmute((*rp).ai_addr),
                                             (*rp).ai_addrlen as uint));
            addrs.push(addrinfo::Info {
                address: addr,
                family: (*rp).ai_family as uint,
                socktype: None,
                protocol: None,
                flags: (*rp).ai_flags as uint
            });

            rp = (*rp).ai_next as *mut libc::addrinfo;
        }
    }

    unsafe { freeaddrinfo(res); }

    Ok(addrs)
}

////////////////////////////////////////////////////////////////////////////////
// get_address_name
////////////////////////////////////////////////////////////////////////////////

extern "system" {
    fn getnameinfo(sa: *const libc::sockaddr, salen: libc::socklen_t,
        host: *mut c_char, hostlen: libc::size_t,
        serv: *mut c_char, servlen: libc::size_t,
        flags: c_int) -> c_int;
}

const NI_MAXHOST: uint = 1025;

pub fn get_address_name(addr: IpAddr) -> Result<String, IoError> {
    let addr = SocketAddr{ip: addr, port: 0};

    let mut storage: libc::sockaddr_storage = unsafe { mem::zeroed() };
    let len = addr_to_sockaddr(addr, &mut storage);

    let mut hostbuf = [0 as c_char; NI_MAXHOST];

    let res = unsafe {
        getnameinfo(&storage as *const _ as *const libc::sockaddr, len,
            hostbuf.as_mut_ptr(), NI_MAXHOST as libc::size_t,
            ptr::null_mut(), 0,
            0)
    };

    if res != 0 {
        return Err(last_gai_error(res));
    }

    unsafe {
        let data = CStr::from_ptr(hostbuf.as_ptr());
        Ok(str::from_utf8(data.to_bytes()).unwrap().to_string())
    }
}

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

pub fn read<T, L, R>(fd: sock_t, deadline: u64, mut lock: L, mut read: R) -> IoResult<uint> where
    L: FnMut() -> T,
    R: FnMut(bool) -> libc::c_int,
{
    let mut ret = -1;
    if deadline == 0 {
        ret = retry(|| read(false));
    }

    if deadline != 0 || (ret == -1 && wouldblock()) {
        let deadline = match deadline {
            0 => None,
            n => Some(n),
        };
        loop {
            // With a timeout, first we wait for the socket to become
            // readable using select(), specifying the relevant timeout for
            // our previously set deadline.
            try!(await(&[fd], deadline, Readable));

            // At this point, we're still within the timeout, and we've
            // determined that the socket is readable (as returned by
            // select). We must still read the socket in *nonblocking* mode
            // because some other thread could come steal our data. If we
            // fail to read some data, we retry (hence the outer loop) and
            // wait for the socket to become readable again.
            let _guard = lock();
            match retry(|| read(deadline.is_some())) {
                -1 if wouldblock() => {}
                -1 => return Err(last_net_error()),
               n => { ret = n; break }
            }
        }
    }

    match ret {
        0 => Err(sys_common::eof()),
        n if n < 0 => Err(last_net_error()),
        n => Ok(n as uint)
    }
}

pub fn write<T, L, W>(fd: sock_t,
                      deadline: u64,
                      buf: &[u8],
                      write_everything: bool,
                      mut lock: L,
                      mut write: W) -> IoResult<uint> where
    L: FnMut() -> T,
    W: FnMut(bool, *const u8, uint) -> i64,
{
    let mut ret = -1;
    let mut written = 0;
    if deadline == 0 {
        if write_everything {
            ret = keep_going(buf, |inner, len| {
                written = buf.len() - len;
                write(false, inner, len)
            });
        } else {
            ret = retry(|| { write(false, buf.as_ptr(), buf.len()) });
            if ret > 0 { written = ret as uint; }
        }
    }

    if deadline != 0 || (ret == -1 && wouldblock()) {
        let deadline = match deadline {
            0 => None,
            n => Some(n),
        };
        while written < buf.len() && (write_everything || written == 0) {
            // As with read(), first wait for the socket to be ready for
            // the I/O operation.
            match await(&[fd], deadline, Writable) {
                Err(ref e) if e.kind == old_io::EndOfFile && written > 0 => {
                    assert!(deadline.is_some());
                    return Err(short_write(written, "short write"))
                }
                Err(e) => return Err(e),
                Ok(()) => {}
            }

            // Also as with read(), we use MSG_DONTWAIT to guard ourselves
            // against unforeseen circumstances.
            let _guard = lock();
            let ptr = buf[written..].as_ptr();
            let len = buf.len() - written;
            match retry(|| write(deadline.is_some(), ptr, len)) {
                -1 if wouldblock() => {}
                -1 => return Err(last_net_error()),
                n => { written += n as uint; }
            }
        }
        ret = 0;
    }
    if ret < 0 {
        Err(last_net_error())
    } else {
        Ok(written)
    }
}

// See http://developerweb.net/viewtopic.php?id=3196 for where this is
// derived from.
pub fn connect_timeout(fd: sock_t,
                       addrp: *const libc::sockaddr,
                       len: libc::socklen_t,
                       timeout_ms: u64) -> IoResult<()> {
    #[cfg(unix)]    use libc::EINPROGRESS as INPROGRESS;
    #[cfg(windows)] use libc::WSAEINPROGRESS as INPROGRESS;
    #[cfg(unix)]    use libc::EWOULDBLOCK as WOULDBLOCK;
    #[cfg(windows)] use libc::WSAEWOULDBLOCK as WOULDBLOCK;

    // Make sure the call to connect() doesn't block
    try!(set_nonblocking(fd, true));

    let ret = match unsafe { libc::connect(fd, addrp, len) } {
        // If the connection is in progress, then we need to wait for it to
        // finish (with a timeout). The current strategy for doing this is
        // to use select() with a timeout.
        -1 if os::errno() as int == INPROGRESS as int ||
              os::errno() as int == WOULDBLOCK as int => {
            let mut set: c::fd_set = unsafe { mem::zeroed() };
            c::fd_set(&mut set, fd);
            match await(fd, &mut set, timeout_ms) {
                0 => Err(timeout("connection timed out")),
                -1 => Err(last_net_error()),
                _ => {
                    let err: libc::c_int = try!(
                        getsockopt(fd, libc::SOL_SOCKET, libc::SO_ERROR));
                    if err == 0 {
                        Ok(())
                    } else {
                        Err(decode_error_detailed(err))
                    }
                }
            }
        }

        -1 => Err(last_net_error()),
        _ => Ok(()),
    };

    // be sure to turn blocking I/O back on
    try!(set_nonblocking(fd, false));
    return ret;

    #[cfg(unix)]
    fn await(fd: sock_t, set: &mut c::fd_set, timeout: u64) -> libc::c_int {
        let start = timer::now();
        retry(|| unsafe {
            // Recalculate the timeout each iteration (it is generally
            // undefined what the value of the 'tv' is after select
            // returns EINTR).
            let mut tv = ms_to_timeval(timeout - (timer::now() - start));
            c::select(fd + 1, ptr::null_mut(), set as *mut _,
                      ptr::null_mut(), &mut tv)
        })
    }
    #[cfg(windows)]
    fn await(_fd: sock_t, set: &mut c::fd_set, timeout: u64) -> libc::c_int {
        let mut tv = ms_to_timeval(timeout);
        unsafe { c::select(1, ptr::null_mut(), set, ptr::null_mut(), &mut tv) }
    }
}

pub fn await(fds: &[sock_t], deadline: Option<u64>,
             status: SocketStatus) -> IoResult<()> {
    let mut set: c::fd_set = unsafe { mem::zeroed() };
    let mut max = 0;
    for &fd in fds {
        c::fd_set(&mut set, fd);
        max = cmp::max(max, fd + 1);
    }
    if cfg!(windows) {
        max = fds.len() as sock_t;
    }

    let (read, write) = match status {
        Readable => (&mut set as *mut _, ptr::null_mut()),
        Writable => (ptr::null_mut(), &mut set as *mut _),
    };
    let mut tv: libc::timeval = unsafe { mem::zeroed() };

    match retry(|| {
        let now = timer::now();
        let tvp = match deadline {
            None => ptr::null_mut(),
            Some(deadline) => {
                // If we're past the deadline, then pass a 0 timeout to
                // select() so we can poll the status
                let ms = if deadline < now {0} else {deadline - now};
                tv = ms_to_timeval(ms);
                &mut tv as *mut _
            }
        };
        let r = unsafe {
            c::select(max as libc::c_int, read, write, ptr::null_mut(), tvp)
        };
        r
    }) {
        -1 => Err(last_net_error()),
        0 => Err(timeout("timed out")),
        _ => Ok(()),
    }
}

////////////////////////////////////////////////////////////////////////////////
// Basic socket representation
////////////////////////////////////////////////////////////////////////////////

struct Inner {
    fd: sock_t,

    // Unused on Linux, where this lock is not necessary.
    #[allow(dead_code)]
    lock: Mutex<()>,
}

impl Inner {
    fn new(fd: sock_t) -> Inner {
        Inner { fd: fd, lock: Mutex::new(()) }
    }
}

impl Drop for Inner {
    fn drop(&mut self) { unsafe { close_sock(self.fd); } }
}

pub struct Guard<'a> {
    pub fd: sock_t,
    pub guard: MutexGuard<'a, ()>,
}

#[unsafe_destructor]
impl<'a> Drop for Guard<'a> {
    fn drop(&mut self) {
        assert!(set_nonblocking(self.fd, false).is_ok());
    }
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
    pub fn connect(addr: SocketAddr, timeout: Option<u64>) -> IoResult<TcpStream> {
        sys::init_net();

        let fd = try!(socket(addr, libc::SOCK_STREAM));
        let ret = TcpStream::new(fd);

        let mut storage = unsafe { mem::zeroed() };
        let len = addr_to_sockaddr(addr, &mut storage);
        let addrp = &storage as *const _ as *const libc::sockaddr;

        match timeout {
            Some(timeout) => {
                try!(connect_timeout(fd, addrp, len, timeout));
                Ok(ret)
            },
            None => {
                match retry(|| unsafe { libc::connect(fd, addrp, len) }) {
                    -1 => Err(last_error()),
                    _ => Ok(ret),
                }
            }
        }
    }

    pub fn new(fd: sock_t) -> TcpStream {
        TcpStream {
            inner: Arc::new(Inner::new(fd)),
            read_deadline: 0,
            write_deadline: 0,
        }
    }

    pub fn fd(&self) -> sock_t { self.inner.fd }

    pub fn set_nodelay(&mut self, nodelay: bool) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_TCP, libc::TCP_NODELAY,
                   nodelay as libc::c_int)
    }

    pub fn set_keepalive(&mut self, seconds: Option<uint>) -> IoResult<()> {
        let ret = setsockopt(self.fd(), libc::SOL_SOCKET, libc::SO_KEEPALIVE,
                             seconds.is_some() as libc::c_int);
        match seconds {
            Some(n) => ret.and_then(|()| self.set_tcp_keepalive(n)),
            None => ret,
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn set_tcp_keepalive(&mut self, seconds: uint) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_TCP, libc::TCP_KEEPALIVE,
                   seconds as libc::c_int)
    }
    #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
    fn set_tcp_keepalive(&mut self, seconds: uint) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_TCP, libc::TCP_KEEPIDLE,
                   seconds as libc::c_int)
    }
    #[cfg(target_os = "openbsd")]
    fn set_tcp_keepalive(&mut self, seconds: uint) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_TCP, libc::SO_KEEPALIVE,
                   seconds as libc::c_int)
    }
    #[cfg(not(any(target_os = "macos",
                  target_os = "ios",
                  target_os = "freebsd",
                  target_os = "dragonfly",
                  target_os = "openbsd")))]
    fn set_tcp_keepalive(&mut self, _seconds: uint) -> IoResult<()> {
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn lock_nonblocking(&self) {}

    #[cfg(not(target_os = "linux"))]
    fn lock_nonblocking<'a>(&'a self) -> Guard<'a> {
        let ret = Guard {
            fd: self.fd(),
            guard: self.inner.lock.lock().unwrap(),
        };
        assert!(set_nonblocking(self.fd(), true).is_ok());
        ret
    }

    pub fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let fd = self.fd();
        let dolock = || self.lock_nonblocking();
        let doread = |nb| unsafe {
            let flags = if nb {c::MSG_DONTWAIT} else {0};
            libc::recv(fd,
                       buf.as_mut_ptr() as *mut libc::c_void,
                       buf.len() as wrlen,
                       flags) as libc::c_int
        };
        read(fd, self.read_deadline, dolock, doread)
    }

    pub fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        let fd = self.fd();
        let dolock = || self.lock_nonblocking();
        let dowrite = |nb: bool, buf: *const u8, len: uint| unsafe {
            let flags = if nb {c::MSG_DONTWAIT} else {0};
            libc::send(fd,
                       buf as *const _,
                       len as wrlen,
                       flags) as i64
        };
        write(fd, self.write_deadline, buf, true, dolock, dowrite).map(|_| ())
    }
    pub fn peer_name(&mut self) -> IoResult<SocketAddr> {
        sockname(self.fd(), libc::getpeername)
    }

    pub fn close_write(&mut self) -> IoResult<()> {
        super::mkerr_libc(unsafe { libc::shutdown(self.fd(), libc::SHUT_WR) })
    }
    pub fn close_read(&mut self) -> IoResult<()> {
        super::mkerr_libc(unsafe { libc::shutdown(self.fd(), libc::SHUT_RD) })
    }

    pub fn set_timeout(&mut self, timeout: Option<u64>) {
        let deadline = timeout.map(|a| timer::now() + a).unwrap_or(0);
        self.read_deadline = deadline;
        self.write_deadline = deadline;
    }
    pub fn set_read_timeout(&mut self, timeout: Option<u64>) {
        self.read_deadline = timeout.map(|a| timer::now() + a).unwrap_or(0);
    }
    pub fn set_write_timeout(&mut self, timeout: Option<u64>) {
        self.write_deadline = timeout.map(|a| timer::now() + a).unwrap_or(0);
    }

    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        sockname(self.fd(), libc::getsockname)
    }
}

impl Clone for TcpStream {
    fn clone(&self) -> TcpStream {
        TcpStream {
            inner: self.inner.clone(),
            read_deadline: 0,
            write_deadline: 0,
        }
    }
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
    pub fn bind(addr: SocketAddr) -> IoResult<UdpSocket> {
        sys::init_net();

        let fd = try!(socket(addr, libc::SOCK_DGRAM));
        let ret = UdpSocket {
            inner: Arc::new(Inner::new(fd)),
            read_deadline: 0,
            write_deadline: 0,
        };

        let mut storage = unsafe { mem::zeroed() };
        let len = addr_to_sockaddr(addr, &mut storage);
        let addrp = &storage as *const _ as *const libc::sockaddr;

        match unsafe { libc::bind(fd, addrp, len) } {
            -1 => Err(last_error()),
            _ => Ok(ret),
        }
    }

    pub fn fd(&self) -> sock_t { self.inner.fd }

    pub fn set_broadcast(&mut self, on: bool) -> IoResult<()> {
        setsockopt(self.fd(), libc::SOL_SOCKET, libc::SO_BROADCAST,
                   on as libc::c_int)
    }

    pub fn set_multicast_loop(&mut self, on: bool) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_IP, libc::IP_MULTICAST_LOOP,
                   on as libc::c_int)
    }

    pub fn set_membership(&mut self, addr: IpAddr, opt: libc::c_int) -> IoResult<()> {
        match ip_to_inaddr(addr) {
            In4Addr(addr) => {
                let mreq = libc::ip_mreq {
                    imr_multiaddr: addr,
                    // interface == INADDR_ANY
                    imr_interface: libc::in_addr { s_addr: 0x0 },
                };
                setsockopt(self.fd(), libc::IPPROTO_IP, opt, mreq)
            }
            In6Addr(addr) => {
                let mreq = libc::ip6_mreq {
                    ipv6mr_multiaddr: addr,
                    ipv6mr_interface: 0,
                };
                setsockopt(self.fd(), libc::IPPROTO_IPV6, opt, mreq)
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn lock_nonblocking(&self) {}

    #[cfg(not(target_os = "linux"))]
    fn lock_nonblocking<'a>(&'a self) -> Guard<'a> {
        let ret = Guard {
            fd: self.fd(),
            guard: self.inner.lock.lock().unwrap(),
        };
        assert!(set_nonblocking(self.fd(), true).is_ok());
        ret
    }

    pub fn socket_name(&mut self) -> IoResult<SocketAddr> {
        sockname(self.fd(), libc::getsockname)
    }

    pub fn recv_from(&mut self, buf: &mut [u8]) -> IoResult<(uint, SocketAddr)> {
        let fd = self.fd();
        let mut storage: libc::sockaddr_storage = unsafe { mem::zeroed() };
        let storagep = &mut storage as *mut _ as *mut libc::sockaddr;
        let mut addrlen: libc::socklen_t =
                mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;

        let dolock = || self.lock_nonblocking();
        let n = try!(read(fd, self.read_deadline, dolock, |nb| unsafe {
            let flags = if nb {c::MSG_DONTWAIT} else {0};
            libc::recvfrom(fd,
                           buf.as_mut_ptr() as *mut libc::c_void,
                           buf.len() as msglen_t,
                           flags,
                           storagep,
                           &mut addrlen) as libc::c_int
        }));
        sockaddr_to_addr(&storage, addrlen as uint).and_then(|addr| {
            Ok((n as uint, addr))
        })
    }

    pub fn send_to(&mut self, buf: &[u8], dst: SocketAddr) -> IoResult<()> {
        let mut storage = unsafe { mem::zeroed() };
        let dstlen = addr_to_sockaddr(dst, &mut storage);
        let dstp = &storage as *const _ as *const libc::sockaddr;

        let fd = self.fd();
        let dolock = || self.lock_nonblocking();
        let dowrite = |nb, buf: *const u8, len: uint| unsafe {
            let flags = if nb {c::MSG_DONTWAIT} else {0};
            libc::sendto(fd,
                         buf as *const libc::c_void,
                         len as msglen_t,
                         flags,
                         dstp,
                         dstlen) as i64
        };

        let n = try!(write(fd, self.write_deadline, buf, false, dolock, dowrite));
        if n != buf.len() {
            Err(short_write(n, "couldn't send entire packet at once"))
        } else {
            Ok(())
        }
    }

    pub fn join_multicast(&mut self, multi: IpAddr) -> IoResult<()> {
        match multi {
            Ipv4Addr(..) => {
                self.set_membership(multi, libc::IP_ADD_MEMBERSHIP)
            }
            Ipv6Addr(..) => {
                self.set_membership(multi, libc::IPV6_ADD_MEMBERSHIP)
            }
        }
    }
    pub fn leave_multicast(&mut self, multi: IpAddr) -> IoResult<()> {
        match multi {
            Ipv4Addr(..) => {
                self.set_membership(multi, libc::IP_DROP_MEMBERSHIP)
            }
            Ipv6Addr(..) => {
                self.set_membership(multi, libc::IPV6_DROP_MEMBERSHIP)
            }
        }
    }

    pub fn multicast_time_to_live(&mut self, ttl: int) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_IP, libc::IP_MULTICAST_TTL,
                   ttl as libc::c_int)
    }
    pub fn time_to_live(&mut self, ttl: int) -> IoResult<()> {
        setsockopt(self.fd(), libc::IPPROTO_IP, libc::IP_TTL, ttl as libc::c_int)
    }

    pub fn set_timeout(&mut self, timeout: Option<u64>) {
        let deadline = timeout.map(|a| timer::now() + a).unwrap_or(0);
        self.read_deadline = deadline;
        self.write_deadline = deadline;
    }
    pub fn set_read_timeout(&mut self, timeout: Option<u64>) {
        self.read_deadline = timeout.map(|a| timer::now() + a).unwrap_or(0);
    }
    pub fn set_write_timeout(&mut self, timeout: Option<u64>) {
        self.write_deadline = timeout.map(|a| timer::now() + a).unwrap_or(0);
    }
}

impl Clone for UdpSocket {
    fn clone(&self) -> UdpSocket {
        UdpSocket {
            inner: self.inner.clone(),
            read_deadline: 0,
            write_deadline: 0,
        }
    }
}
