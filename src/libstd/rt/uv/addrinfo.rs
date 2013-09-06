// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cast::transmute;
use cell::Cell;
use c_str::ToCStr;
use libc::{c_int, c_void};
use option::{Option, Some, None};
use ptr::null;
use rt::uv::uvll;
use rt::uv::uvll::UV_GETADDRINFO;
use rt::uv::{Loop, UvError, NativeHandle};
use rt::uv::status_to_maybe_uv_error_with_loop;
use rt::uv::net::UvAddrInfo;

type GetAddrInfoCallback = ~fn(GetAddrInfoRequest, &UvAddrInfo, Option<UvError>);

pub struct GetAddrInfoRequest(*uvll::uv_getaddrinfo_t);

pub struct RequestData {
    getaddrinfo_cb: Option<GetAddrInfoCallback>,
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
                       service: Option<&str>, hints: Option<UvAddrInfo>,
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

        // XXX: Implement hints
        assert!(hints.is_none());

        self.get_req_data().getaddrinfo_cb = Some(wrapper_cb);

        unsafe {
            assert!(0 == uvll::getaddrinfo(loop_.native_handle(),
                                           self.native_handle(),
                                           getaddrinfo_cb,
                                           c_node_ptr,
                                           c_service_ptr,
                                           null()));
        }

        extern "C" fn getaddrinfo_cb(req: *uvll::uv_getaddrinfo_t,
                                     status: c_int,
                                     res: *uvll::addrinfo) {
            let mut req: GetAddrInfoRequest = NativeHandle::from_native_handle(req);
            let loop_ = req.get_loop();
            let err = status_to_maybe_uv_error_with_loop(loop_.native_handle(), status);
            let addrinfo = UvAddrInfo(res);
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
    use option::{Some, None};
    use rt::uv::Loop;
    use rt::uv::net::accum_sockaddrs;
    use rt::io::net::ip::{SocketAddr, Ipv4Addr};
    use super::*;

    #[test]
    fn getaddrinfo_test() {
        let mut loop_ = Loop::new();
        let mut req = GetAddrInfoRequest::new();
        do req.getaddrinfo(&loop_, Some("localhost"), None, None) |_, addrinfo, _| {
            let sockaddrs = accum_sockaddrs(addrinfo);
            let mut found_local = false;
            let local_addr = &SocketAddr {
                ip: Ipv4Addr(127, 0, 0, 1),
                port: 0
            };
            for addr in sockaddrs.iter() {
                found_local = found_local || addr == local_addr;
            }
            assert!(found_local);
        }
        loop_.run();
        loop_.close();
        req.delete();
    }
}
