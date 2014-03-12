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
use std::iter::Iterator;
use std::libc;
use std::mem;
use std::ptr;
use std::result::Result;
use std::str::raw;
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

pub fn sockaddr_to_network_addr(sa: *libc::sockaddr, useLocal: bool) -> Option<~NetworkAddress> {
    unsafe {
        if (*sa).sa_family as libc::c_int == libc::AF_PACKET {
            let sll: *libc::sockaddr_ll = cast::transmute(sa);
            let ni = if useLocal {
                let nis = get_network_interfaces();
                if nis.iter().filter(|x| x.index as i32 == (*sll).sll_ifindex).len() == 1 {
                    (*nis.iter().filter(|x| x.index as i32 == (*sll).sll_ifindex).next().unwrap()).clone()
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
                //sll.sll_addr = [a, b, c, d, e, f, 0, 0];
                (storage, mem::size_of::<libc::sockaddr_ll>())
            }
            //MacAddress(MacAddr(a, b, c, d, e, f)) => {
            //    let mut storage: libc::sockaddr_storage = mem::init();
            //    let sll: &mut libc::sockaddr_ll = cast::transmute(&mut storage);
            //    sll.sll_family = libc::AF_PACKET as libc::sa_family_t;
            //    sll.sll_addr = [a, b, c, d, e, f, 0, 0];
            //    (storage, mem::size_of::<libc::sockaddr_ll>())
            //}
        }
    }
}

pub fn sockaddr_to_network_addrs(sa: *libc::sockaddr)
    -> (Option<MacAddr>, Option<IpAddr>, Option<IpAddr>) {
    //(None, None, None)
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
            let name = raw::from_c_str((*addr).ifa_name);
            let (mac, ipv4, ipv6) = sockaddr_to_network_addrs((*addr).ifa_addr);
            let ni = ~NetworkInterface {
                name: name.clone(),
                index: 0,
                mac: mac,
                ipv4: ipv4,
                ipv6: ipv6,
                flags: (*addr).ifa_flags
            };
            //println!("name: {:?}; mac: {:?}; ipv4: {:?}; ipv6: {:?};", name, mac, ipv4, ipv6);
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

