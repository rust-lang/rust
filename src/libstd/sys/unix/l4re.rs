// Copyright 2016-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! unimpl {
    () => (return Err(io::Error::new(io::ErrorKind::Other, "No networking available on L4Re."));)
}

pub mod net {
    #![allow(warnings)]
    use fmt;
    use io;
    use libc;
    use net::{SocketAddr, Shutdown, Ipv4Addr, Ipv6Addr};
    use sys_common::{AsInner, FromInner, IntoInner};
    use sys::fd::FileDesc;
    use time::Duration;


    pub extern crate libc as netc;

    pub struct Socket(FileDesc);
    impl Socket {
        pub fn new(_: &SocketAddr, _: libc::c_int) -> io::Result<Socket> {
            unimpl!();
        }

        pub fn new_raw(_: libc::c_int, _: libc::c_int) -> io::Result<Socket> {
            unimpl!();
        }

        pub fn new_pair(_: libc::c_int, _: libc::c_int) -> io::Result<(Socket, Socket)> {
            unimpl!();
        }

        pub fn connect_timeout(&self, _: &SocketAddr, _: Duration) -> io::Result<()> {
            unimpl!();
        }

        pub fn accept(&self, _: *mut libc::sockaddr, _: *mut libc::socklen_t)
                  -> io::Result<Socket> {
            unimpl!();
        }

        pub fn duplicate(&self) -> io::Result<Socket> {
            unimpl!();
        }

        pub fn read(&self, _: &mut [u8]) -> io::Result<usize> {
            unimpl!();
        }

        pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
            unimpl!();
        }

        pub fn recv_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
            unimpl!();
        }

        pub fn peek_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
            unimpl!();
        }

        pub fn write(&self, _: &[u8]) -> io::Result<usize> {
            unimpl!();
        }

        pub fn set_timeout(&self, _: Option<Duration>, _: libc::c_int) -> io::Result<()> {
            unimpl!();
        }

        pub fn timeout(&self, _: libc::c_int) -> io::Result<Option<Duration>> {
            unimpl!();
        }

        pub fn shutdown(&self, _: Shutdown) -> io::Result<()> {
            unimpl!();
        }

        pub fn set_nodelay(&self, _: bool) -> io::Result<()> {
            unimpl!();
        }

        pub fn nodelay(&self) -> io::Result<bool> {
            unimpl!();
        }

        pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
            unimpl!();
        }

        pub fn take_error(&self) -> io::Result<Option<io::Error>> {
            unimpl!();
        }
    }

    impl AsInner<libc::c_int> for Socket {
        fn as_inner(&self) -> &libc::c_int { self.0.as_inner() }
    }

    impl FromInner<libc::c_int> for Socket {
        fn from_inner(fd: libc::c_int) -> Socket { Socket(FileDesc::new(fd)) }
    }

    impl IntoInner<libc::c_int> for Socket {
        fn into_inner(self) -> libc::c_int { self.0.into_raw() }
    }

    pub struct TcpStream {
        inner: Socket,
    }

    impl TcpStream {
        pub fn connect(_: &SocketAddr) -> io::Result<TcpStream> {
            unimpl!();
        }

        pub fn connect_timeout(_: &SocketAddr, _: Duration) -> io::Result<TcpStream> {
            unimpl!();
        }

        pub fn socket(&self) -> &Socket { &self.inner }

        pub fn into_socket(self) -> Socket { self.inner }

        pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
            unimpl!();
        }

        pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
            unimpl!();
        }

        pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
            unimpl!();
        }

        pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
            unimpl!();
        }

        pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
            unimpl!();
        }

        pub fn read(&self, _: &mut [u8]) -> io::Result<usize> {
            unimpl!();
        }

        pub fn write(&self, _: &[u8]) -> io::Result<usize> {
            unimpl!();
        }

        pub fn peer_addr(&self) -> io::Result<SocketAddr> {
            unimpl!();
        }

        pub fn socket_addr(&self) -> io::Result<SocketAddr> {
            unimpl!();
        }

        pub fn shutdown(&self, _: Shutdown) -> io::Result<()> {
            unimpl!();
        }

        pub fn duplicate(&self) -> io::Result<TcpStream> {
            unimpl!();
        }

        pub fn set_nodelay(&self, _: bool) -> io::Result<()> {
            unimpl!();
        }

        pub fn nodelay(&self) -> io::Result<bool> {
            unimpl!();
        }

        pub fn set_ttl(&self, _: u32) -> io::Result<()> {
            unimpl!();
        }

        pub fn ttl(&self) -> io::Result<u32> {
            unimpl!();
        }

        pub fn take_error(&self) -> io::Result<Option<io::Error>> {
            unimpl!();
        }

        pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
            unimpl!();
        }
    }

    impl FromInner<Socket> for TcpStream {
        fn from_inner(socket: Socket) -> TcpStream {
            TcpStream { inner: socket }
        }
    }

    impl fmt::Debug for TcpStream {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "No networking support available on L4Re")
        }
    }

    pub struct TcpListener {
        inner: Socket,
    }

    impl TcpListener {
        pub fn bind(_: &SocketAddr) -> io::Result<TcpListener> {
            unimpl!();
        }

        pub fn socket(&self) -> &Socket { &self.inner }

        pub fn into_socket(self) -> Socket { self.inner }

        pub fn socket_addr(&self) -> io::Result<SocketAddr> {
            unimpl!();
        }

        pub fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
            unimpl!();
        }

        pub fn duplicate(&self) -> io::Result<TcpListener> {
            unimpl!();
        }

        pub fn set_ttl(&self, _: u32) -> io::Result<()> {
            unimpl!();
        }

        pub fn ttl(&self) -> io::Result<u32> {
            unimpl!();
        }

        pub fn set_only_v6(&self, _: bool) -> io::Result<()> {
            unimpl!();
        }

        pub fn only_v6(&self) -> io::Result<bool> {
            unimpl!();
        }

        pub fn take_error(&self) -> io::Result<Option<io::Error>> {
            unimpl!();
        }

        pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
            unimpl!();
        }
    }

    impl FromInner<Socket> for TcpListener {
        fn from_inner(socket: Socket) -> TcpListener {
            TcpListener { inner: socket }
        }
    }

    impl fmt::Debug for TcpListener {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "No networking support available on L4Re.")
        }
    }

    pub struct UdpSocket {
        inner: Socket,
    }

    impl UdpSocket {
        pub fn bind(_: &SocketAddr) -> io::Result<UdpSocket> {
            unimpl!();
        }

        pub fn socket(&self) -> &Socket { &self.inner }

        pub fn into_socket(self) -> Socket { self.inner }

        pub fn socket_addr(&self) -> io::Result<SocketAddr> {
            unimpl!();
        }

        pub fn recv_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
            unimpl!();
        }

        pub fn peek_from(&self, _: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
            unimpl!();
        }

        pub fn send_to(&self, _: &[u8], _: &SocketAddr) -> io::Result<usize> {
            unimpl!();
        }

        pub fn duplicate(&self) -> io::Result<UdpSocket> {
            unimpl!();
        }

        pub fn set_read_timeout(&self, _: Option<Duration>) -> io::Result<()> {
            unimpl!();
        }

        pub fn set_write_timeout(&self, _: Option<Duration>) -> io::Result<()> {
            unimpl!();
        }

        pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
            unimpl!();
        }

        pub fn write_timeout(&self) -> io::Result<Option<Duration>> {
            unimpl!();
        }

        pub fn set_broadcast(&self, _: bool) -> io::Result<()> {
            unimpl!();
        }

        pub fn broadcast(&self) -> io::Result<bool> {
            unimpl!();
        }

        pub fn set_multicast_loop_v4(&self, _: bool) -> io::Result<()> {
            unimpl!();
        }

        pub fn multicast_loop_v4(&self) -> io::Result<bool> {
            unimpl!();
        }

        pub fn set_multicast_ttl_v4(&self, _: u32) -> io::Result<()> {
            unimpl!();
        }

        pub fn multicast_ttl_v4(&self) -> io::Result<u32> {
            unimpl!();
        }

        pub fn set_multicast_loop_v6(&self, _: bool) -> io::Result<()> {
            unimpl!();
        }

        pub fn multicast_loop_v6(&self) -> io::Result<bool> {
            unimpl!();
        }

        pub fn join_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr)
                             -> io::Result<()> {
                                 unimpl!();
        }

        pub fn join_multicast_v6(&self, _: &Ipv6Addr, _: u32)
                             -> io::Result<()> {
                                 unimpl!();
        }

        pub fn leave_multicast_v4(&self, _: &Ipv4Addr, _: &Ipv4Addr)
                              -> io::Result<()> {
                                  unimpl!();
        }

        pub fn leave_multicast_v6(&self, _: &Ipv6Addr, _: u32)
                              -> io::Result<()> {
                                  unimpl!();
        }

        pub fn set_ttl(&self, _: u32) -> io::Result<()> {
            unimpl!();
        }

        pub fn ttl(&self) -> io::Result<u32> {
            unimpl!();
        }

        pub fn take_error(&self) -> io::Result<Option<io::Error>> {
            unimpl!();
        }

        pub fn set_nonblocking(&self, _: bool) -> io::Result<()> {
            unimpl!();
        }

        pub fn recv(&self, _: &mut [u8]) -> io::Result<usize> {
            unimpl!();
        }

        pub fn peek(&self, _: &mut [u8]) -> io::Result<usize> {
            unimpl!();
        }

        pub fn send(&self, _: &[u8]) -> io::Result<usize> {
            unimpl!();
        }

        pub fn connect(&self, _: &SocketAddr) -> io::Result<()> {
            unimpl!();
        }
    }

    impl FromInner<Socket> for UdpSocket {
        fn from_inner(socket: Socket) -> UdpSocket {
            UdpSocket { inner: socket }
        }
    }

    impl fmt::Debug for UdpSocket {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "No networking support on L4Re available.")
        }
    }

    pub struct LookupHost {
        original: *mut libc::addrinfo,
        cur: *mut libc::addrinfo,
    }

    impl Iterator for LookupHost {
        type Item = SocketAddr;
        fn next(&mut self) -> Option<SocketAddr> {
            None
        }
    }

    unsafe impl Sync for LookupHost {}
    unsafe impl Send for LookupHost {}

    pub fn lookup_host(_: &str) -> io::Result<LookupHost> {
        unimpl!();
    }

    pub fn res_init_if_glibc_before_2_26() -> io::Result<()> {
        unimpl!();
    }
}

