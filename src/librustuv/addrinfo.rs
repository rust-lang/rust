// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cast::transmute;
use std::cell::Cell;
use std::libc::{c_int, c_void};
use std::ptr::null;
use ai = std::rt::io::net::addrinfo;

use uvll;
use uvll::UV_GETADDRINFO;
use super::{Loop, UvError, NativeHandle, status_to_maybe_uv_error};
use net;

type GetAddrInfoCallback = ~fn(GetAddrInfoRequest, &net::UvAddrInfo, Option<UvError>);

pub struct GetAddrInfoRequest(*uvll::uv_getaddrinfo_t);

pub struct RequestData {
    priv getaddrinfo_cb: Option<GetAddrInfoCallback>,
}

impl GetAddrInfoRequest {
    pub fn new() -> GetAddrInfoRequest {
        let req = unsafe { uvll::malloc_req(UV_GETADDRINFO) };
        assert!(req.is_not_null());
        let mut req: GetAddrInfoRequest = NativeHandle::from_native_handle(req);
        req.install_req_data();
        return req;
    }

    pub fn getaddrinfo(&mut self, loop_: &Loop, node: Option<&str>,
                       service: Option<&str>, hints: Option<ai::Hint>,
                       cb: GetAddrInfoCallback) {

        assert!(node.is_some() || service.is_some());

        let (c_node, c_node_ptr) = match node {
            Some(n) => {
                let c_node = n.to_c_str();
                let c_node_ptr = c_node.with_ref(|r| r);
                (Some(c_node), c_node_ptr)
            }
            None => (None, null())
        };

        let (c_service, c_service_ptr) = match service {
            Some(s) => {
                let c_service = s.to_c_str();
                let c_service_ptr = c_service.with_ref(|r| r);
                (Some(c_service), c_service_ptr)
            }
            None => (None, null())
        };

        let cb = Cell::new(cb);
        let wrapper_cb: GetAddrInfoCallback = |req, addrinfo, err| {
            // Capture some heap values that need to stay alive for the
            // getaddrinfo call
            let _ = &c_node;
            let _ = &c_service;

            let cb = cb.take();
            cb(req, addrinfo, err)
        };

        let hint = hints.map(|hint| {
            let mut flags = 0;
            do each_ai_flag |cval, aival| {
                if hint.flags & (aival as uint) != 0 {
                    flags |= cval as i32;
                }
            }
            /* XXX: do we really want to support these?
            let socktype = match hint.socktype {
                Some(ai::Stream) => uvll::rust_SOCK_STREAM(),
                Some(ai::Datagram) => uvll::rust_SOCK_DGRAM(),
                Some(ai::Raw) => uvll::rust_SOCK_RAW(),
                None => 0,
            };
            let protocol = match hint.protocol {
                Some(ai::UDP) => uvll::rust_IPPROTO_UDP(),
                Some(ai::TCP) => uvll::rust_IPPROTO_TCP(),
                _ => 0,
            };
            */
            let socktype = 0;
            let protocol = 0;

            uvll::addrinfo {
                ai_flags: flags,
                ai_family: hint.family as c_int,
                ai_socktype: socktype,
                ai_protocol: protocol,
                ai_addrlen: 0,
                ai_canonname: null(),
                ai_addr: null(),
                ai_next: null(),
            }
        });
        let hint_ptr = hint.as_ref().map_default(null(), |x| x as *uvll::addrinfo);

        self.get_req_data().getaddrinfo_cb = Some(wrapper_cb);

        unsafe {
            assert!(0 == uvll::getaddrinfo(loop_.native_handle(),
                                           self.native_handle(),
                                           getaddrinfo_cb,
                                           c_node_ptr,
                                           c_service_ptr,
                                           hint_ptr));
        }

        extern "C" fn getaddrinfo_cb(req: *uvll::uv_getaddrinfo_t,
                                     status: c_int,
                                     res: *uvll::addrinfo) {
            let mut req: GetAddrInfoRequest = NativeHandle::from_native_handle(req);
            let err = status_to_maybe_uv_error(status);
            let addrinfo = net::UvAddrInfo(res);
            let data = req.get_req_data();
            (*data.getaddrinfo_cb.get_ref())(req, &addrinfo, err);
            unsafe {
                uvll::freeaddrinfo(res);
            }
        }
    }

    fn get_loop(&self) -> Loop {
        unsafe {
            Loop {
                handle: uvll::get_loop_from_fs_req(self.native_handle())
            }
        }
    }

    fn install_req_data(&mut self) {
        let req = self.native_handle() as *uvll::uv_getaddrinfo_t;
        let data = ~RequestData {
            getaddrinfo_cb: None
        };
        unsafe {
            let data = transmute::<~RequestData, *c_void>(data);
            uvll::set_data_for_req(req, data);
        }
    }

    fn get_req_data<'r>(&'r mut self) -> &'r mut RequestData {
        unsafe {
            let data = uvll::get_data_for_req(self.native_handle());
            let data = transmute::<&*c_void, &mut ~RequestData>(&data);
            return &mut **data;
        }
    }

    fn delete(self) {
        unsafe {
            let data = uvll::get_data_for_req(self.native_handle());
            let _data = transmute::<*c_void, ~RequestData>(data);
            uvll::set_data_for_req(self.native_handle(), null::<()>());
            uvll::free_req(self.native_handle());
        }
    }
}

fn each_ai_flag(_f: &fn(c_int, ai::Flag)) {
    /* XXX: do we really want to support these?
    unsafe {
        f(uvll::rust_AI_ADDRCONFIG(), ai::AddrConfig);
        f(uvll::rust_AI_ALL(), ai::All);
        f(uvll::rust_AI_CANONNAME(), ai::CanonName);
        f(uvll::rust_AI_NUMERICHOST(), ai::NumericHost);
        f(uvll::rust_AI_NUMERICSERV(), ai::NumericServ);
        f(uvll::rust_AI_PASSIVE(), ai::Passive);
        f(uvll::rust_AI_V4MAPPED(), ai::V4Mapped);
    }
    */
}

// Traverse the addrinfo linked list, producing a vector of Rust socket addresses
pub fn accum_addrinfo(addr: &net::UvAddrInfo) -> ~[ai::Info] {
    unsafe {
        let &net::UvAddrInfo(addr) = addr;
        let mut addr = addr;

        let mut addrs = ~[];
        loop {
            let uvaddr = net::sockaddr_to_UvSocketAddr((*addr).ai_addr);
            let rustaddr = net::uv_socket_addr_to_socket_addr(uvaddr);

            let mut flags = 0;
            do each_ai_flag |cval, aival| {
                if (*addr).ai_flags & cval != 0 {
                    flags |= aival as uint;
                }
            }

            /* XXX: do we really want to support these
            let protocol = match (*addr).ai_protocol {
                p if p == uvll::rust_IPPROTO_UDP() => Some(ai::UDP),
                p if p == uvll::rust_IPPROTO_TCP() => Some(ai::TCP),
                _ => None,
            };
            let socktype = match (*addr).ai_socktype {
                p if p == uvll::rust_SOCK_STREAM() => Some(ai::Stream),
                p if p == uvll::rust_SOCK_DGRAM() => Some(ai::Datagram),
                p if p == uvll::rust_SOCK_RAW() => Some(ai::Raw),
                _ => None,
            };
            */
            let protocol = None;
            let socktype = None;

            addrs.push(ai::Info {
                address: rustaddr,
                family: (*addr).ai_family as uint,
                socktype: socktype,
                protocol: protocol,
                flags: flags,
            });
            if (*addr).ai_next.is_not_null() {
                addr = (*addr).ai_next;
            } else {
                break;
            }
        }

        return addrs;
    }
}

impl NativeHandle<*uvll::uv_getaddrinfo_t> for GetAddrInfoRequest {
    fn from_native_handle(handle: *uvll::uv_getaddrinfo_t) -> GetAddrInfoRequest {
        GetAddrInfoRequest(handle)
    }
    fn native_handle(&self) -> *uvll::uv_getaddrinfo_t {
        match self { &GetAddrInfoRequest(ptr) => ptr }
    }
}

#[cfg(test)]
mod test {
    use Loop;
    use std::rt::io::net::ip::{SocketAddr, Ipv4Addr};
    use super::*;

    #[test]
    fn getaddrinfo_test() {
        let mut loop_ = Loop::new();
        let mut req = GetAddrInfoRequest::new();
        do req.getaddrinfo(&loop_, Some("localhost"), None, None) |_, addrinfo, _| {
            let sockaddrs = accum_addrinfo(addrinfo);
            let mut found_local = false;
            let local_addr = &SocketAddr {
                ip: Ipv4Addr(127, 0, 0, 1),
                port: 0
            };
            for addr in sockaddrs.iter() {
                found_local = found_local || addr.address == *local_addr;
            }
            assert!(found_local);
        }
        loop_.run();
        loop_.close();
        req.delete();
    }
}
