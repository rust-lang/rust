// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use inner::prelude::*;

use core::hash;
use libc::{self, socklen_t, sa_family_t};
use core::mem;
use super::{ntoh, hton};
use net::{self as sys, IpAddr, SocketAddr};
use super::ip::{Ipv4Addr, Ipv6Addr};
use core::option;

#[derive(Copy)]
pub struct SocketAddrV4 { inner: libc::sockaddr_in }

#[derive(Copy)]
pub struct SocketAddrV6 { inner: libc::sockaddr_in6 }

impl sys::SocketAddrV4 for SocketAddrV4 {
    type Addr = Ipv4Addr;

    fn new(ip: Ipv4Addr, port: u16) -> SocketAddrV4 {
        SocketAddrV4 {
            inner: libc::sockaddr_in {
                sin_family: libc::AF_INET as sa_family_t,
                sin_port: hton(port),
                sin_addr: *ip.as_inner(),
                .. unsafe { mem::zeroed() }
            },
        }
    }

    fn addr(&self) -> &Ipv4Addr {
        unsafe {
            &*(&self.inner.sin_addr as *const libc::in_addr as *const Ipv4Addr)
        }
    }

    fn port(&self) -> u16 { ntoh(self.inner.sin_port) }
}

impl sys::SocketAddrV6 for SocketAddrV6 {
    type Addr = Ipv6Addr;

    fn new(ip: Ipv6Addr, port: u16, flowinfo: u32, scope_id: u32)
               -> SocketAddrV6 {
        SocketAddrV6 {
            inner: libc::sockaddr_in6 {
                sin6_family: libc::AF_INET6 as sa_family_t,
                sin6_port: hton(port),
                sin6_addr: *ip.as_inner(),
                sin6_flowinfo: hton(flowinfo),
                sin6_scope_id: hton(scope_id),
                .. unsafe { mem::zeroed() }
            },
        }
    }

    fn addr(&self) -> &Ipv6Addr {
        unsafe {
            &*(&self.inner.sin6_addr as *const libc::in6_addr as *const Ipv6Addr)
        }
    }

    fn port(&self) -> u16 { ntoh(self.inner.sin6_port) }

    fn flowinfo(&self) -> u32 { ntoh(self.inner.sin6_flowinfo) }

    fn scope_id(&self) -> u32 { ntoh(self.inner.sin6_scope_id) }
}

impl Clone for SocketAddrV4 {
    fn clone(&self) -> SocketAddrV4 { *self }
}

impl Clone for SocketAddrV6 {
    fn clone(&self) -> SocketAddrV6 { *self }
}

impl PartialEq for SocketAddrV4 {
    fn eq(&self, other: &SocketAddrV4) -> bool {
        self.inner.sin_port == other.inner.sin_port &&
            self.inner.sin_addr.s_addr == other.inner.sin_addr.s_addr
    }
}

impl PartialEq for SocketAddrV6 {
    fn eq(&self, other: &SocketAddrV6) -> bool {
        self.inner.sin6_port == other.inner.sin6_port &&
            self.inner.sin6_addr.s6_addr == other.inner.sin6_addr.s6_addr &&
            self.inner.sin6_flowinfo == other.inner.sin6_flowinfo &&
            self.inner.sin6_scope_id == other.inner.sin6_scope_id
    }
}

impl Eq for SocketAddrV4 {}
impl Eq for SocketAddrV6 {}

impl hash::Hash for SocketAddrV4 {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        (self.inner.sin_port, self.inner.sin_addr.s_addr).hash(s)
    }
}

impl hash::Hash for SocketAddrV6 {
    fn hash<H: hash::Hasher>(&self, s: &mut H) {
        (self.inner.sin6_port, &self.inner.sin6_addr.s6_addr,
         self.inner.sin6_flowinfo, self.inner.sin6_scope_id).hash(s)
    }
}

impl AsInner<libc::sockaddr_in> for SocketAddrV4 {
    fn as_inner(&self) -> &libc::sockaddr_in {
        &self.inner
    }
}

impl IntoInner<libc::sockaddr_in> for SocketAddrV4 {
    fn into_inner(self) -> libc::sockaddr_in {
        self.inner
    }
}

impl FromInner<libc::sockaddr_in> for SocketAddrV4 {
    fn from_inner(inner: libc::sockaddr_in) -> Self {
        SocketAddrV4 { inner: inner }
    }
}

impl AsInner<libc::sockaddr_in6> for SocketAddrV6 {
    fn as_inner(&self) -> &libc::sockaddr_in6 {
        &self.inner
    }
}

impl IntoInner<libc::sockaddr_in6> for SocketAddrV6 {
    fn into_inner(self) -> libc::sockaddr_in6 {
        self.inner
    }
}

impl FromInner<libc::sockaddr_in6> for SocketAddrV6 {
    fn from_inner(inner: libc::sockaddr_in6) -> Self {
        SocketAddrV6 { inner: inner }
    }
}

pub fn sockaddr(addr: &SocketAddr<super::Net>) -> (*const libc::sockaddr, socklen_t) {
    match *addr {
        SocketAddr::V4(ref a) => {
            (&a.inner as *const _ as *const _, mem::size_of_val(&a.inner) as socklen_t)
        }
        SocketAddr::V6(ref a) => {
            (&a.inner as *const _ as *const _, mem::size_of_val(&a.inner) as socklen_t)
        }
    }
}

pub fn new_sockaddr<N: sys::Net>(addr: IpAddr<N>, port: u16) -> SocketAddr<N> {
    match addr {
        IpAddr::V4(a) => SocketAddr::V4(<N::SocketAddrV4 as sys::SocketAddrV4>::new(a, port)),
        IpAddr::V6(a) => SocketAddr::V6(<N::SocketAddrV6 as sys::SocketAddrV6>::new(a, port, 0, 0)),
    }
}
