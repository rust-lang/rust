// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cell::UnsafeCell;
use cmp;
use io::{Error, ErrorKind, Result};
use mem;
use net::{SocketAddr, Ipv4Addr, Ipv6Addr};
use path::Path;
use sys::fs::{File, OpenOptions};
use sys::syscall::TimeSpec;
use sys_common::{AsInner, FromInner, IntoInner};
use time::Duration;

use super::{path_to_peer_addr, path_to_local_addr};

#[derive(Debug)]
pub struct UdpSocket(File, UnsafeCell<Option<SocketAddr>>);

impl UdpSocket {
    pub fn bind(addr: &SocketAddr) -> Result<UdpSocket> {
        let path = format!("udp:/{}", addr);
        let mut options = OpenOptions::new();
        options.read(true);
        options.write(true);
        Ok(UdpSocket(File::open(&Path::new(path.as_str()), &options)?, UnsafeCell::new(None)))
    }

    fn get_conn(&self) -> &mut Option<SocketAddr> {
        unsafe { &mut *(self.1.get()) }
    }

    pub fn connect(&self, addr: &SocketAddr) -> Result<()> {
        unsafe { *self.1.get() = Some(*addr) };
        Ok(())
    }

    pub fn duplicate(&self) -> Result<UdpSocket> {
        let new_bind = self.0.dup(&[])?;
        let new_conn = *self.get_conn();
        Ok(UdpSocket(new_bind, UnsafeCell::new(new_conn)))
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)> {
        let from = self.0.dup(b"listen")?;
        let path = from.path()?;
        let peer_addr = path_to_peer_addr(path.to_str().unwrap_or(""));
        let count = from.read(buf)?;
        Ok((count, peer_addr))
    }

    pub fn recv(&self, buf: &mut [u8]) -> Result<usize> {
        if let Some(addr) = *self.get_conn() {
            let from = self.0.dup(format!("{}", addr).as_bytes())?;
            from.read(buf)
        } else {
            Err(Error::new(ErrorKind::Other, "UdpSocket::recv not connected"))
        }
    }

    pub fn send_to(&self, buf: &[u8], addr: &SocketAddr) -> Result<usize> {
        let to = self.0.dup(format!("{}", addr).as_bytes())?;
        to.write(buf)
    }

    pub fn send(&self, buf: &[u8]) -> Result<usize> {
        if let Some(addr) = *self.get_conn() {
            self.send_to(buf, &addr)
        } else {
            Err(Error::new(ErrorKind::Other, "UdpSocket::send not connected"))
        }
    }

    pub fn take_error(&self) -> Result<Option<Error>> {
        Ok(None)
    }

    pub fn socket_addr(&self) -> Result<SocketAddr> {
        let path = self.0.path()?;
        Ok(path_to_local_addr(path.to_str().unwrap_or("")))
    }

    pub fn broadcast(&self) -> Result<bool> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::broadcast not implemented"))
    }

    pub fn multicast_loop_v4(&self) -> Result<bool> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::multicast_loop_v4 not implemented"))
    }

    pub fn multicast_loop_v6(&self) -> Result<bool> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::multicast_loop_v6 not implemented"))
    }

    pub fn multicast_ttl_v4(&self) -> Result<u32> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::multicast_ttl_v4 not implemented"))
    }

    pub fn nonblocking(&self) -> Result<bool> {
        self.0.fd().nonblocking()
    }

    pub fn only_v6(&self) -> Result<bool> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::only_v6 not implemented"))
    }

    pub fn ttl(&self) -> Result<u32> {
        let mut ttl = [0];
        let file = self.0.dup(b"ttl")?;
        file.read(&mut ttl)?;
        Ok(ttl[0] as u32)
    }

    pub fn read_timeout(&self) -> Result<Option<Duration>> {
        let mut time = TimeSpec::default();
        let file = self.0.dup(b"read_timeout")?;
        if file.read(&mut time)? >= mem::size_of::<TimeSpec>() {
            Ok(Some(Duration::new(time.tv_sec as u64, time.tv_nsec as u32)))
        } else {
            Ok(None)
        }
    }

    pub fn write_timeout(&self) -> Result<Option<Duration>> {
        let mut time = TimeSpec::default();
        let file = self.0.dup(b"write_timeout")?;
        if file.read(&mut time)? >= mem::size_of::<TimeSpec>() {
            Ok(Some(Duration::new(time.tv_sec as u64, time.tv_nsec as u32)))
        } else {
            Ok(None)
        }
    }

    pub fn set_broadcast(&self, _broadcast: bool) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::set_broadcast not implemented"))
    }

    pub fn set_multicast_loop_v4(&self, _multicast_loop_v4: bool) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::set_multicast_loop_v4 not implemented"))
    }

    pub fn set_multicast_loop_v6(&self, _multicast_loop_v6: bool) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::set_multicast_loop_v6 not implemented"))
    }

    pub fn set_multicast_ttl_v4(&self, _multicast_ttl_v4: u32) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::set_multicast_ttl_v4 not implemented"))
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> Result<()> {
        self.0.fd().set_nonblocking(nonblocking)
    }

    pub fn set_only_v6(&self, _only_v6: bool) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::set_only_v6 not implemented"))
    }

    pub fn set_ttl(&self, ttl: u32) -> Result<()> {
        let file = self.0.dup(b"ttl")?;
        file.write(&[cmp::min(ttl, 255) as u8])?;
        Ok(())
    }

    pub fn set_read_timeout(&self, duration_option: Option<Duration>) -> Result<()> {
        let file = self.0.dup(b"read_timeout")?;
        if let Some(duration) = duration_option {
            file.write(&TimeSpec {
                tv_sec: duration.as_secs() as i64,
                tv_nsec: duration.subsec_nanos() as i32
            })?;
        } else {
            file.write(&[])?;
        }
        Ok(())
    }

    pub fn set_write_timeout(&self, duration_option: Option<Duration>) -> Result<()> {
        let file = self.0.dup(b"write_timeout")?;
        if let Some(duration) = duration_option {
            file.write(&TimeSpec {
                tv_sec: duration.as_secs() as i64,
                tv_nsec: duration.subsec_nanos() as i32
            })?;
        } else {
            file.write(&[])?;
        }
        Ok(())
    }

    pub fn join_multicast_v4(&self, _multiaddr: &Ipv4Addr, _interface: &Ipv4Addr) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::join_multicast_v4 not implemented"))
    }

    pub fn join_multicast_v6(&self, _multiaddr: &Ipv6Addr, _interface: u32) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::join_multicast_v6 not implemented"))
    }

    pub fn leave_multicast_v4(&self, _multiaddr: &Ipv4Addr, _interface: &Ipv4Addr) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::leave_multicast_v4 not implemented"))
    }

    pub fn leave_multicast_v6(&self, _multiaddr: &Ipv6Addr, _interface: u32) -> Result<()> {
        Err(Error::new(ErrorKind::Other, "UdpSocket::leave_multicast_v6 not implemented"))
    }
}

impl AsInner<File> for UdpSocket {
    fn as_inner(&self) -> &File { &self.0 }
}

impl FromInner<File> for UdpSocket {
    fn from_inner(file: File) -> UdpSocket {
        UdpSocket(file, UnsafeCell::new(None))
    }
}

impl IntoInner<File> for UdpSocket {
    fn into_inner(self) -> File { self.0 }
}
