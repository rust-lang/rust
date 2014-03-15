// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Networking support for code common to both librustuv and libnative

#[crate_id = "netsupport#0.10-pre"];
#[license = "MIT/ASL2"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];
#[doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://static.rust-lang.org/doc/master")];

use std::cast;
use std::io;
use std::io::net::ip;
use std::io::net::ip::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::io::net::raw::{IpAddress, MacAddr, NetworkAddress, NetworkInterface};
use std::io::net::raw;
use std::iter::Iterator;
use std::libc;
use std::mem;
use std::os;
use std::ptr;
use std::result::Result;
use strraw = std::str::raw;
use std::vec_ng::Vec;

pub fn htons(u: u16) -> u16 {
    mem::to_be16(u as i16) as u16
}
pub fn ntohs(u: u16) -> u16 {
    mem::from_be16(u as i16) as u16
}

pub enum InAddr {
    InAddr(libc::in_addr),
    In6Addr(libc::in6_addr),
}

pub fn ip_to_inaddr(ip: ip::IpAddr) -> InAddr {
    match ip {
        ip::Ipv4Addr(a, b, c, d) => {
            InAddr(libc::in_addr {
                s_addr: (d as u32 << 24) |
                        (c as u32 << 16) |
                        (b as u32 <<  8) |
                        (a as u32 <<  0)
            })
        }
        ip::Ipv6Addr(a, b, c, d, e, f, g, h) => {
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

pub fn addr_to_sockaddr(addr: ip::SocketAddr) -> (libc::sockaddr_storage, uint) {
    unsafe {
        let storage: libc::sockaddr_storage = mem::init();
        let len = match ip_to_inaddr(addr.ip) {
            InAddr(inaddr) => {
                let storage: *mut libc::sockaddr_in = cast::transmute(&storage);
                (*storage).sin_family = libc::AF_INET as libc::sa_family_t;
                (*storage).sin_port = htons(addr.port);
                (*storage).sin_addr = inaddr;
                mem::size_of::<libc::sockaddr_in>()
            }
            In6Addr(inaddr) => {
                let storage: *mut libc::sockaddr_in6 = cast::transmute(&storage);
                (*storage).sin6_family = libc::AF_INET6 as libc::sa_family_t;
                (*storage).sin6_port = htons(addr.port);
                (*storage).sin6_addr = inaddr;
                mem::size_of::<libc::sockaddr_in6>()
            }
        };
        return (storage, len);
    }
}

pub fn sockaddr_to_addr(storage: &libc::sockaddr_storage,
                        len: uint) -> Result<ip::SocketAddr, io::IoError> {
    match storage.ss_family as libc::c_int {
        libc::AF_INET => {
            assert!(len as uint >= mem::size_of::<libc::sockaddr_in>());
            let storage: &libc::sockaddr_in = unsafe {
                cast::transmute(storage)
            };
            let addr = storage.sin_addr.s_addr as u32;
            let a = (addr >>  0) as u8;
            let b = (addr >>  8) as u8;
            let c = (addr >> 16) as u8;
            let d = (addr >> 24) as u8;
            Ok(ip::SocketAddr {
                ip: ip::Ipv4Addr(a, b, c, d),
                port: ntohs(storage.sin_port),
            })
        }
        libc::AF_INET6 => {
            assert!(len as uint >= mem::size_of::<libc::sockaddr_in6>());
            let storage: &libc::sockaddr_in6 = unsafe {
                cast::transmute(storage)
            };
            let a = ntohs(storage.sin6_addr.s6_addr[0]);
            let b = ntohs(storage.sin6_addr.s6_addr[1]);
            let c = ntohs(storage.sin6_addr.s6_addr[2]);
            let d = ntohs(storage.sin6_addr.s6_addr[3]);
            let e = ntohs(storage.sin6_addr.s6_addr[4]);
            let f = ntohs(storage.sin6_addr.s6_addr[5]);
            let g = ntohs(storage.sin6_addr.s6_addr[6]);
            let h = ntohs(storage.sin6_addr.s6_addr[7]);
            Ok(ip::SocketAddr {
                ip: ip::Ipv6Addr(a, b, c, d, e, f, g, h),
                port: ntohs(storage.sin6_port),
            })
        }
        _ => {
            Err(io::standard_error(io::OtherIoError))
        }
    }
}

#[cfg(target_os = "linux")]
pub fn sockaddr_to_network_addr(sa: *libc::sockaddr, useLocal: bool) -> Option<~NetworkAddress> {
    unsafe {
        if (*sa).sa_family as libc::c_int == libc::AF_PACKET {
            let sll: *libc::sockaddr_ll = cast::transmute(sa);
            let ni = if useLocal {
                let nis = get_network_interfaces();
                if nis.iter().filter(|x| x.index as i32 == (*sll).sll_ifindex).len() == 1 {
                    (*nis.iter()
                         .filter(|x| x.index as i32 == (*sll).sll_ifindex)
                         .next()
                         .unwrap())
                    .clone()
                } else {
                    sll_to_ni(*sll)
                }
            } else {
                sll_to_ni(*sll)
            };

            return Some(~NetworkAddress(ni));
        } else {
            return Some(~IpAddress(sockaddr_to_addr(cast::transmute(sa),
                                                    mem::size_of::<libc::sockaddr_storage>()
                                                    ).unwrap().ip));
        }
    }

    fn sll_to_ni(sll: libc::sockaddr_ll) -> ~NetworkInterface {
        let mac = MacAddr(sll.sll_addr[0], sll.sll_addr[1],
                          sll.sll_addr[2], sll.sll_addr[3],
                          sll.sll_addr[4], sll.sll_addr[5]);
        ~NetworkInterface {
            name: ~"",
            index: 0,
            mac: Some(mac),
            ipv4: None,
            ipv6: None,
            flags: 0
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn sockaddr_to_network_addr(sa: *libc::sockaddr, _useLocal: bool) -> Option<~NetworkAddress> {
    unsafe {
        Some(
            ~IpAddress(
                sockaddr_to_addr(cast::transmute(sa),
                     mem::size_of::<libc::sockaddr_storage>()
                ).unwrap().ip)
        )
    }
}


#[cfg(target_os = "linux")]
pub fn network_addr_to_sockaddr(na: ~NetworkAddress) -> (libc::sockaddr_storage, uint) {
    unsafe {
        match na {
            ~IpAddress(ip) => addr_to_sockaddr(ip::SocketAddr { ip: ip, port : 0}),
            //_ => (mem::init(), 0)
            ~NetworkAddress(ni) => {
                let mut storage: libc::sockaddr_storage = mem::init();
                let sll: &mut libc::sockaddr_ll = cast::transmute(&mut storage);
                sll.sll_family = libc::AF_PACKET as libc::sa_family_t;
                match ni.mac {
                    Some(MacAddr(a, b, c, d, e, f)) => sll.sll_addr = [a, b, c, d, e, f, 0, 0],
                    _ => ()
                }
                sll.sll_halen = 6;
                sll.sll_ifindex = ni.index as i32;
                (storage, mem::size_of::<libc::sockaddr_ll>())
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn network_addr_to_sockaddr(na: ~NetworkAddress) -> (libc::sockaddr_storage, uint) {
     match na {
         ~IpAddress(ip) => addr_to_sockaddr(ip::SocketAddr { ip: ip, port : 0}),
         _ => fail!("Layer 2 networking not supported on this OS")
     }
}

pub fn sockaddr_to_network_addrs(sa: *libc::sockaddr)
    -> (Option<MacAddr>, Option<IpAddr>, Option<IpAddr>) {
    match sockaddr_to_network_addr(sa, false) {
        Some(~IpAddress(ip@Ipv4Addr(..))) => (None, Some(ip), None),
        Some(~IpAddress(ip@Ipv6Addr(..))) => (None, None, Some(ip)),
        Some(~NetworkAddress(ni)) => (ni.mac, None, None),
        None => (None, None, None)
    }
}

pub fn get_network_interfaces() -> Vec<~NetworkInterface> {
    let mut ifaces: Vec<~NetworkInterface> = Vec::new();
    unsafe {
        let mut addrs: *libc::ifaddrs = mem::init();
        if libc::getifaddrs(&mut addrs) != 0 {
            return ifaces;
        }
        let mut addr = addrs;
        while addr != ptr::null() {
            let name = strraw::from_c_str((*addr).ifa_name);
            let (mac, ipv4, ipv6) = sockaddr_to_network_addrs((*addr).ifa_addr);
            let ni = ~NetworkInterface {
                name: name.clone(),
                index: 0,
                mac: mac,
                ipv4: ipv4,
                ipv6: ipv6,
                flags: (*addr).ifa_flags
            };
            let mut found: bool = false;
            for iface in ifaces.mut_iter() {
                if name == iface.name {
                    merge(iface, &ni);
                    found = true;
                }
            }
            if !found {
                ifaces.push(ni);
            }

            addr = (*addr).ifa_next;
        }
        libc::freeifaddrs(addrs);

        for iface in ifaces.mut_iter() {
            iface.index = libc::if_nametoindex(iface.name.to_c_str().unwrap());
        }
        return ifaces;
    }

    fn merge(old: &mut ~NetworkInterface, new: &~NetworkInterface) {
        old.mac = match new.mac {
            None => old.mac,
            _ => new.mac
        };
        old.ipv4 = match new.ipv4 {
            None => old.ipv4,
            _ => new.ipv4
        };
        old.ipv6 = match new.ipv6 {
            None => old.ipv6,
            _ => new.ipv6
        };
        old.flags = old.flags | new.flags;
    }

}

#[cfg(target_os = "linux")]
pub fn protocol_to_libc(protocol: raw::Protocol)
    -> (libc::c_int, libc::c_int, libc::c_int) {
    let eth_p_all: u16 = htons(0x0003);
    match protocol {
        raw::DataLinkProtocol(raw::EthernetProtocol)
            => (libc::AF_PACKET, libc::SOCK_RAW, eth_p_all as libc::c_int),
        raw::DataLinkProtocol(raw::CookedEthernetProtocol(proto))
            => (libc::AF_PACKET, libc::SOCK_DGRAM, proto as libc::c_int),
        raw::NetworkProtocol(raw::Ipv4NetworkProtocol)
            => (libc::AF_INET, libc::SOCK_RAW, libc::IPPROTO_RAW),
        raw::NetworkProtocol(raw::Ipv6NetworkProtocol)
            => (libc::AF_INET6, libc::SOCK_RAW, libc::IPPROTO_RAW),
        raw::TransportProtocol(raw::Ipv4TransportProtocol(proto))
            => (libc::AF_INET, libc::SOCK_RAW, proto as libc::c_int),
        raw::TransportProtocol(raw::Ipv6TransportProtocol(proto))
            => (libc::AF_INET6, libc::SOCK_RAW, proto as libc::c_int)
    }
}

#[cfg(not(target_os = "linux"))]
pub fn protocol_to_libc(protocol: raw::Protocol)
    -> (libc::c_int, libc::c_int, libc::c_int) {
    match protocol {
        raw::NetworkProtocol(raw::Ipv4NetworkProtocol)
            => (libc::AF_INET, libc::SOCK_RAW, libc::IPPROTO_RAW),
        raw::NetworkProtocol(raw::Ipv6NetworkProtocol)
            => (libc::AF_INET6, libc::SOCK_RAW, libc::IPPROTO_RAW),
        raw::TransportProtocol(raw::Ipv4TransportProtocol(proto))
            => (libc::AF_INET, libc::SOCK_RAW, proto as libc::c_int),
        raw::TransportProtocol(raw::Ipv6TransportProtocol(proto))
            => (libc::AF_INET6, libc::SOCK_RAW, proto as libc::c_int),
        _   => fail!("Layer 2 networking not supported on this OS")
    }
}

pub fn translate_error(errno: i32, detail: bool) -> io::IoError {
    #[cfg(windows)]
    fn get_err(errno: i32) -> (io::IoErrorKind, &'static str) {
        match errno {
            libc::EOF => (io::EndOfFile, "end of file"),
            libc::ERROR_NO_DATA => (io::BrokenPipe, "the pipe is being closed"),
            libc::ERROR_FILE_NOT_FOUND => (io::FileNotFound, "file not found"),
            libc::ERROR_INVALID_NAME => (io::InvalidInput, "invalid file name"),
            libc::WSAECONNREFUSED => (io::ConnectionRefused, "connection refused"),
            libc::WSAECONNRESET => (io::ConnectionReset, "connection reset"),
            libc::WSAEACCES => (io::PermissionDenied, "permission denied"),
            libc::WSAEWOULDBLOCK =>
                (io::ResourceUnavailable, "resource temporarily unavailable"),
            libc::WSAENOTCONN => (io::NotConnected, "not connected"),
            libc::WSAECONNABORTED => (io::ConnectionAborted, "connection aborted"),
            libc::WSAEADDRNOTAVAIL => (io::ConnectionRefused, "address not available"),
            libc::WSAEADDRINUSE => (io::ConnectionRefused, "address in use"),
            libc::ERROR_BROKEN_PIPE => (io::EndOfFile, "the pipe has ended"),

            // libuv maps this error code to EISDIR. we do too. if it is found
            // to be incorrect, we can add in some more machinery to only
            // return this message when ERROR_INVALID_FUNCTION after certain
            // win32 calls.
            libc::ERROR_INVALID_FUNCTION => (io::InvalidInput,
                                             "illegal operation on a directory"),

            x => {
                debug!("ignoring {}: {}", x, os::last_os_error());
                (io::OtherIoError, "unknown error")
            }
        }
    }

    #[cfg(not(windows))]
    fn get_err(errno: i32) -> (io::IoErrorKind, &'static str) {
        // FIXME: this should probably be a bit more descriptive...
        match errno {
            libc::EOF => (io::EndOfFile, "end of file"),
            libc::ECONNREFUSED => (io::ConnectionRefused, "connection refused"),
            libc::ECONNRESET => (io::ConnectionReset, "connection reset"),
            libc::EPERM | libc::EACCES =>
                (io::PermissionDenied, "permission denied"),
            libc::EPIPE => (io::BrokenPipe, "broken pipe"),
            libc::ENOTCONN => (io::NotConnected, "not connected"),
            libc::ECONNABORTED => (io::ConnectionAborted, "connection aborted"),
            libc::EADDRNOTAVAIL => (io::ConnectionRefused, "address not available"),
            libc::EADDRINUSE => (io::ConnectionRefused, "address in use"),
            libc::ENOENT => (io::FileNotFound, "no such file or directory"),
            libc::EISDIR => (io::InvalidInput, "illegal operation on a directory"),

            // These two constants can have the same value on some systems, but
            // different values on others, so we can't use a match clause
            x if x == libc::EAGAIN || x == libc::EWOULDBLOCK =>
                (io::ResourceUnavailable, "resource temporarily unavailable"),

            x => {
                debug!("ignoring {}: {}", x, os::last_os_error());
                (io::OtherIoError, "unknown error")
            }
        }
    }

    let (kind, desc) = get_err(errno);
    io::IoError {
        kind: kind,
        desc: desc,
        detail: if detail {Some(os::last_os_error())} else {None},
    }
}

pub fn last_error() -> io::IoError { translate_error(os::errno() as i32, true) }
