// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use sys::inner::*;

use hash;
use libc::{self, socklen_t, sa_family_t};
use mem;
use super::{ntoh, hton};
use sys::common::net::{SocketAddr, IpAddr};
use sys::net as sys;
use super::ip::{IpAddrV4, IpAddrV6};

#[derive(Copy)]
pub struct SocketAddrV4(libc::sockaddr_in);

#[derive(Copy)]
pub struct SocketAddrV6(libc::sockaddr_in6);

impl SocketAddrV4 {
    pub fn new(ip: IpAddrV4, port: u16) -> SocketAddrV4 {
        SocketAddrV4(
            libc::sockaddr_in {
                sin_family: libc::AF_INET as sa_family_t,
                sin_port: hton(port),
                sin_addr: *ip.as_inner(),
                .. unsafe { mem::zeroed() }
            }
        )
    }

    pub fn addr(&self) -> &IpAddrV4 {
        unsafe {
            &*(&self.0.sin_addr as *const libc::in_addr as *const IpAddrV4)
        }
    }

    pub fn port(&self) -> u16 { ntoh(self.0.sin_port) }
}

impl SocketAddrV6 {
    pub fn new(ip: IpAddrV6, port: u16, flowinfo: u32, scope_id: u32)
               -> SocketAddrV6 {
        SocketAddrV6(
            libc::sockaddr_in6 {
                sin6_family: libc::AF_INET6 as sa_family_t,
                sin6_port: hton(port),
                sin6_addr: *ip.as_inner(),
                sin6_flowinfo: hton(flowinfo),
                sin6_scope_id: hton(scope_id),
                .. unsafe { mem::zeroed() }
            }
        )
    }

    pub fn addr(&self) -> &IpAddrV6 {
        unsafe {
            &*(&self.0.sin6_addr as *const libc::in6_addr as *const IpAddrV6)
        }
    }

    pub fn port(&self) -> u16 { ntoh(self.0.sin6_port) }

    pub fn flowinfo(&self) -> u32 { ntoh(self.0.sin6_flowinfo) }

    pub fn scope_id(&self) -> u32 { ntoh(self.0.sin6_scope_id) }
}

impl Clone for SocketAddrV4 {
    fn clone(&self) -> SocketAddrV4 { *self }
}

impl Clone for SocketAddrV6 {
    fn clone(&self) -> SocketAddrV6 { *self }
}

impl PartialEq for SocketAddrV4 {
    fn eq(&self, other: &SocketAddrV4) -> bool {
        self.0.sin_port == other.0.sin_port &&
            self.0.sin_addr.s_addr == other.0.sin_addr.s_addr
    }
}

impl PartialEq for SocketAddrV6 {
    fn eq(&self, other: &SocketAddrV6) -> bool {
        self.0.sin6_port == other.0.sin6_port &&
            self.0.sin6_addr.s6_addr == other.0.sin6_addr.s6_addr &&
            self.0.sin6_flowinfo == other.0.sin6_flowinfo &&
            self.0.sin6_scope_id == other.0.sin6_scope_id
    }
}

impl Eq for SocketAddrV4 {}
impl Eq for SocketAddrV6 {}

impl hash::Hash for SocketAddrV4 {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        (self.0.sin_port, self.0.sin_addr.s_addr).hash(s)
    }
}

impl hash::Hash for SocketAddrV6 {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        (self.0.sin6_port, &self.0.sin6_addr.s6_addr,
         self.0.sin6_flowinfo, self.0.sin6_scope_id).hash(s)
    }
}

impl_inner!(SocketAddrV4(libc::sockaddr_in));
impl_inner!(SocketAddrV6(libc::sockaddr_in6));

pub fn sockaddr(addr: &SocketAddr) -> (*const libc::sockaddr, socklen_t) {
    match *addr {
        SocketAddr::V4(ref a) => {
            (&a.0 as *const _ as *const _, mem::size_of_val(&a.0) as socklen_t)
        }
        SocketAddr::V6(ref a) => {
            (&a.0 as *const _ as *const _, mem::size_of_val(&a.0) as socklen_t)
        }
    }
}

pub fn new_sockaddr(addr: IpAddr, port: u16) -> SocketAddr {
    match addr {
        IpAddr::V4(a) => SocketAddr::V4(sys::SocketAddrV4::new(a, port)),
        IpAddr::V6(a) => SocketAddr::V6(sys::SocketAddrV6::new(a, port, 0, 0)),
    }
}
