// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::{c_char, c_int};
use libc;
use std::mem;
use std::ptr::{null, null_mut};
use std::rt::rtio;
use std::rt::rtio::IoError;

use super::net;

pub struct GetAddrInfoRequest;

impl GetAddrInfoRequest {
    pub fn run(host: Option<&str>, servname: Option<&str>,
               hint: Option<rtio::AddrinfoHint>)
        -> Result<Vec<rtio::AddrinfoInfo>, IoError>
    {
        assert!(host.is_some() || servname.is_some());

        let c_host = host.map(|x| x.to_c_str());
        let c_host = c_host.as_ref().map(|x| x.as_ptr()).unwrap_or(null());
        let c_serv = servname.map(|x| x.to_c_str());
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
            return Err(get_error(s));
        }

        // Collect all the results we found
        let mut addrs = Vec::new();
        let mut rp = res;
        while rp.is_not_null() {
            unsafe {
                let addr = match net::sockaddr_to_addr(mem::transmute((*rp).ai_addr),
                                                       (*rp).ai_addrlen as uint) {
                    Ok(a) => a,
                    Err(e) => return Err(e)
                };
                addrs.push(rtio::AddrinfoInfo {
                    address: addr,
                    family: (*rp).ai_family as uint,
                    socktype: 0,
                    protocol: 0,
                    flags: (*rp).ai_flags as uint
                });

                rp = (*rp).ai_next as *mut libc::addrinfo;
            }
        }

        unsafe { freeaddrinfo(res); }

        Ok(addrs)
    }
}

extern "system" {
    fn getaddrinfo(node: *const c_char, service: *const c_char,
                   hints: *const libc::addrinfo,
                   res: *mut *mut libc::addrinfo) -> c_int;
    fn freeaddrinfo(res: *mut libc::addrinfo);
    #[cfg(not(windows))]
    fn gai_strerror(errcode: c_int) -> *const c_char;
}

#[cfg(windows)]
fn get_error(_: c_int) -> IoError {
    net::last_error()
}

#[cfg(not(windows))]
fn get_error(s: c_int) -> IoError {
    use std::c_str::CString;

    let err_str = unsafe {
        CString::new(gai_strerror(s), false).as_str().unwrap().to_string()
    };
    IoError {
        code: s as uint,
        extra: 0,
        detail: Some(err_str),
    }
}
