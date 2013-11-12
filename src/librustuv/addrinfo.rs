// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ai = std::io::net::addrinfo;
use std::libc::c_int;
use std::ptr::null;
use std::rt::BlockedTask;
use std::rt::local::Local;
use std::rt::sched::Scheduler;

use net;
use super::{Loop, UvError, Request, wait_until_woken_after};
use uvll;

struct Addrinfo {
    handle: *uvll::addrinfo,
}

struct Ctx {
    slot: Option<BlockedTask>,
    status: c_int,
    addrinfo: Option<Addrinfo>,
}

pub struct GetAddrInfoRequest;

impl GetAddrInfoRequest {
    pub fn run(loop_: &Loop, node: Option<&str>, service: Option<&str>,
               hints: Option<ai::Hint>) -> Result<~[ai::Info], UvError> {
        assert!(node.is_some() || service.is_some());
        let (_c_node, c_node_ptr) = match node {
            Some(n) => {
                let c_node = n.to_c_str();
                let c_node_ptr = c_node.with_ref(|r| r);
                (Some(c_node), c_node_ptr)
            }
            None => (None, null())
        };

        let (_c_service, c_service_ptr) = match service {
            Some(s) => {
                let c_service = s.to_c_str();
                let c_service_ptr = c_service.with_ref(|r| r);
                (Some(c_service), c_service_ptr)
            }
            None => (None, null())
        };

        let hint = hints.map(|hint| {
            let mut flags = 0;
            do each_ai_flag |cval, aival| {
                if hint.flags & (aival as uint) != 0 {
                    flags |= cval as i32;
                }
            }
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
        let mut req = Request::new(uvll::UV_GETADDRINFO);

        return match unsafe {
            uvll::uv_getaddrinfo(loop_.handle, req.handle,
                                 getaddrinfo_cb, c_node_ptr, c_service_ptr,
                                 hint_ptr)
        } {
            0 => {
                req.defuse(); // uv callback now owns this request
                let mut cx = Ctx { slot: None, status: 0, addrinfo: None };

                do wait_until_woken_after(&mut cx.slot) {
                    req.set_data(&cx);
                }

                match cx.status {
                    0 => Ok(accum_addrinfo(cx.addrinfo.get_ref())),
                    n => Err(UvError(n))
                }
            }
            n => Err(UvError(n))
        };


        extern fn getaddrinfo_cb(req: *uvll::uv_getaddrinfo_t,
                                 status: c_int,
                                 res: *uvll::addrinfo) {
            let req = Request::wrap(req);
            assert!(status != uvll::ECANCELED);
            let cx: &mut Ctx = unsafe { req.get_data() };
            cx.status = status;
            cx.addrinfo = Some(Addrinfo { handle: res });

            let sched: ~Scheduler = Local::take();
            sched.resume_blocked_task_immediately(cx.slot.take_unwrap());
        }
    }
}

impl Drop for Addrinfo {
    fn drop(&mut self) {
        unsafe { uvll::uv_freeaddrinfo(self.handle) }
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
pub fn accum_addrinfo(addr: &Addrinfo) -> ~[ai::Info] {
    unsafe {
        let mut addr = addr.handle;

        let mut addrs = ~[];
        loop {
            let rustaddr = net::sockaddr_to_socket_addr((*addr).ai_addr);

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

#[cfg(test)]
mod test {
    use std::io::net::ip::{SocketAddr, Ipv4Addr};
    use super::*;
    use super::super::local_loop;

    #[test]
    #[ignore(cfg(target_os="android"))] // cannot give tcp/ip permission without help of apk
    fn getaddrinfo_test() {
        match GetAddrInfoRequest::run(local_loop(), Some("localhost"), None, None) {
            Ok(infos) => {
                let mut found_local = false;
                let local_addr = &SocketAddr {
                    ip: Ipv4Addr(127, 0, 0, 1),
                    port: 0
                };
                for addr in infos.iter() {
                    found_local = found_local || addr.address == *local_addr;
                }
                assert!(found_local);
            }
            Err(e) => fail!("{:?}", e),
        }
    }
}
