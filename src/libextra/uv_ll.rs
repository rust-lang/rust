// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Low-level bindings to the libuv library.
 *
 * This module contains a set of direct, 'bare-metal' wrappers around
 * the libuv C-API.
 *
 * Also contained herein are a set of rust records that map, in
 * approximate memory-size, to the libuv data structures. The record
 * implementations are adjusted, per-platform, to match their respective
 * representations.
 *
 * There are also a collection of helper functions to ease interacting
 * with the low-level API (such as a function to return the latest
 * libuv error as a rust-formatted string).
 *
 * As new functionality, existant in uv.h, is added to the rust stdlib,
 * the mappings should be added in this module.
 *
 * This module's implementation will hopefully be, eventually, replaced
 * with per-platform, generated source files from rust-bindgen.
 */

#[allow(non_camel_case_types)]; // C types

use core::prelude::*;

use core::libc::size_t;
use core::libc::c_void;
use core::ptr::to_unsafe_ptr;

pub type uv_handle_t = c_void;
pub type uv_loop_t = c_void;
pub type uv_idle_t = c_void;
pub type uv_idle_cb = *u8;

// libuv struct mappings
pub struct uv_ip4_addr {
    ip: ~[u8],
    port: int,
}
pub type uv_ip6_addr = uv_ip4_addr;

pub enum uv_handle_type {
    UNKNOWN_HANDLE = 0,
    UV_TCP,
    UV_UDP,
    UV_NAMED_PIPE,
    UV_TTY,
    UV_FILE,
    UV_TIMER,
    UV_PREPARE,
    UV_CHECK,
    UV_IDLE,
    UV_ASYNC,
    UV_ARES_TASK,
    UV_ARES_EVENT,
    UV_PROCESS,
    UV_FS_EVENT
}

pub type handle_type = libc::c_uint;

pub struct uv_handle_fields {
   loop_handle: *libc::c_void,
   type_: handle_type,
   close_cb: *u8,
   data: *libc::c_void,
}

// unix size: 8
pub struct uv_err_t {
    code: libc::c_int,
    sys_errno_: libc::c_int
}

// don't create one of these directly. instead,
// count on it appearing in libuv callbacks or embedded
// in other types as a pointer to be used in other
// operations (so mostly treat it as opaque, once you
// have it in this form..)
pub struct uv_stream_t {
    fields: uv_handle_fields,
}

// 64bit unix size: 216
#[cfg(target_os="macos")]
pub struct uv_tcp_t {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8, a11: *u8,
    a12: *u8, a13: *u8, a14: *u8, a15: *u8,
    a16: *u8, a17: *u8, a18: *u8, a19: *u8,
    a20: *u8, a21: *u8, a22: *u8,
    a23: uv_tcp_t_osx_riders
}
#[cfg(target_arch="x86_64")]
pub struct uv_tcp_t_osx_riders {
    a23: *u8,
}
#[cfg(target_arch="x86")]
#[cfg(target_arch="arm")]
pub struct uv_tcp_t_osx_riders {
    a23: *u8,
    a24: *u8, a25: *u8,
}
#[cfg(target_os="linux")]
#[cfg(target_os="freebsd")]
#[cfg(target_os="android")]
pub struct uv_tcp_t {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8, a11: *u8,
    a12: *u8, a13: *u8, a14: *u8, a15: *u8,
    a16: *u8, a17: *u8, a18: *u8, a19: *u8,
    a20: *u8, a21: *u8,
    a22: uv_tcp_t_32bit_unix_riders,
}
// 32bit unix size: 328 (164)
#[cfg(target_arch="x86_64")]
pub struct uv_tcp_t_32bit_unix_riders {
    a29: *u8,
}
#[cfg(target_arch="x86")]
#[cfg(target_arch="arm")]
#[cfg(target_arch="mips")]
pub struct uv_tcp_t_32bit_unix_riders {
    a29: *u8, a30: *u8, a31: *u8,
}

// 32bit win32 size: 240 (120)
#[cfg(windows)]
pub struct uv_tcp_t {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8, a11: *u8,
    a12: *u8, a13: *u8, a14: *u8, a15: *u8,
    a16: *u8, a17: *u8, a18: *u8, a19: *u8,
    a20: *u8, a21: *u8, a22: *u8, a23: *u8,
    a24: *u8, a25: *u8,
}

// unix size: 64
#[cfg(unix)]
pub struct uv_connect_t {
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8
}
// win32 size: 88 (44)
#[cfg(windows)]
pub struct uv_connect_t {
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8,
}

// unix size: 16
pub struct uv_buf_t {
    base: *u8,
    len: libc::size_t,
}
// no gen stub method.. should create
// it via uv::direct::buf_init()

// unix size: 160
#[cfg(unix)]
pub struct uv_write_t {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8, a11: *u8,
    a12: *u8,
    a14: uv_write_t_32bit_unix_riders,
}
#[cfg(target_arch="x86_64")]
pub struct uv_write_t_32bit_unix_riders {
    a13: *u8, a14: *u8, a15: *u8
}
#[cfg(target_arch="x86")]
#[cfg(target_arch="arm")]
#[cfg(target_arch="mips")]
pub struct uv_write_t_32bit_unix_riders {
    a13: *u8, a14: *u8, a15: *u8,
    a16: *u8,
}
// win32 size: 136 (68)
#[cfg(windows)]
pub struct uv_write_t {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8, a11: *u8,
    a12: *u8,
}
// 64bit unix size: 96
// 32bit unix size: 152 (76)
#[cfg(unix)]
pub struct uv_async_t {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8,
    a07: uv_async_t_32bit_unix_riders,
}
#[cfg(target_arch="x86_64")]
pub struct uv_async_t_32bit_unix_riders {
    a10: *u8,
}
#[cfg(target_arch="x86")]
#[cfg(target_arch="arm")]
#[cfg(target_arch="mips")]
pub struct uv_async_t_32bit_unix_riders {
    a10: *u8,
}
// win32 size 132 (68)
#[cfg(windows)]
pub struct uv_async_t {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8, a11: *u8,
    a12: *u8,
}

// 64bit unix size: 120
// 32bit unix size: 84
#[cfg(unix)]
pub struct uv_timer_t {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8,
    a11: uv_timer_t_32bit_unix_riders,
}
#[cfg(target_arch="x86_64")]
pub struct uv_timer_t_32bit_unix_riders {
    a10: *u8,
}
#[cfg(target_arch="x86")]
#[cfg(target_arch="arm")]
#[cfg(target_arch="mips")]
pub struct uv_timer_t_32bit_unix_riders {
    a10: *u8, a11: *u8, a12: *u8
}
// win32 size: 64
#[cfg(windows)]
pub struct uv_timer_t {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8, a11: *u8,
}

// unix size: 16
pub struct sockaddr_in {
    sin_family: u16,
    sin_port: u16,
    sin_addr: u32, // in_addr: this is an opaque, per-platform struct
    sin_zero: (u8, u8, u8, u8, u8, u8, u8, u8),
}

// unix size: 28 .. FIXME #1645
// stuck with 32 because of rust padding structs?
#[cfg(target_arch="x86_64")]
pub struct sockaddr_in6 {
    a0: *u8, a1: *u8,
    a2: *u8, a3: *u8,
}
#[cfg(target_arch="x86")]
#[cfg(target_arch="arm")]
#[cfg(target_arch="mips")]
pub struct sockaddr_in6 {
    a0: *u8, a1: *u8,
    a2: *u8, a3: *u8,
    a4: *u8, a5: *u8,
    a6: *u8, a7: *u8,
}

// unix size: 28 .. FIXME #1645
// stuck with 32 because of rust padding structs?
pub type addr_in = addr_in_impl::addr_in;
#[cfg(unix)]
pub mod addr_in_impl {
    #[cfg(target_arch="x86_64")]
    pub struct addr_in {
        a0: *u8, a1: *u8,
        a2: *u8, a3: *u8,
    }
    #[cfg(target_arch="x86")]
    #[cfg(target_arch="arm")]
    #[cfg(target_arch="mips")]
    pub struct addr_in {
        a0: *u8, a1: *u8,
        a2: *u8, a3: *u8,
        a4: *u8, a5: *u8,
        a6: *u8, a7: *u8,
    }
}
#[cfg(windows)]
pub mod addr_in_impl {
    pub struct addr_in {
        a0: *u8, a1: *u8,
        a2: *u8, a3: *u8,
    }
}

// unix size: 48, 32bit: 32
pub type addrinfo = addrinfo_impl::addrinfo;
#[cfg(target_os="linux")]
#[cfg(target_os="android")]
pub mod addrinfo_impl {
    #[cfg(target_arch="x86_64")]
    pub struct addrinfo {
        a00: *u8, a01: *u8, a02: *u8, a03: *u8,
        a04: *u8, a05: *u8,
    }
    #[cfg(target_arch="x86")]
    #[cfg(target_arch="arm")]
    #[cfg(target_arch="mips")]
    pub struct addrinfo {
        a00: *u8, a01: *u8, a02: *u8, a03: *u8,
        a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    }
}
#[cfg(target_os="macos")]
#[cfg(target_os="freebsd")]
pub mod addrinfo_impl {
    pub struct addrinfo {
        a00: *u8, a01: *u8, a02: *u8, a03: *u8,
        a04: *u8, a05: *u8,
    }
}
#[cfg(windows)]
pub mod addrinfo_impl {
    pub struct addrinfo {
        a00: *u8, a01: *u8, a02: *u8, a03: *u8,
        a04: *u8, a05: *u8,
    }
}

// unix size: 72
pub struct uv_getaddrinfo_t {
    a00: *u8, a01: *u8, a02: *u8, a03: *u8, a04: *u8, a05: *u8,
    a06: *u8, a07: *u8, a08: *u8, a09: *u8,
    a10: *u8, a11: *u8, a12: *u8, a13: *u8, a14: *u8, a15: *u8
}

pub mod uv_ll_struct_stubgen {

    use core::ptr;

    use super::{
        uv_async_t,
        uv_connect_t,
        uv_getaddrinfo_t,
        uv_handle_fields,
        uv_tcp_t,
        uv_timer_t,
        uv_write_t,
    };

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    use super::{
        uv_async_t_32bit_unix_riders,
        uv_timer_t_32bit_unix_riders,
        uv_write_t_32bit_unix_riders,
    };

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    #[cfg(target_os = "freebsd")]
    use super::uv_tcp_t_32bit_unix_riders;

    pub fn gen_stub_uv_tcp_t() -> uv_tcp_t {
        return gen_stub_os();
        #[cfg(target_os = "linux")]
        #[cfg(target_os = "android")]
        #[cfg(target_os = "freebsd")]
        pub fn gen_stub_os() -> uv_tcp_t {
            return gen_stub_arch();
            #[cfg(target_arch="x86_64")]
            pub fn gen_stub_arch() -> uv_tcp_t {
                uv_tcp_t {
                    fields: uv_handle_fields {
                        loop_handle: ptr::null(), type_: 0u32,
                        close_cb: ptr::null(),
                        data: ptr::null(),
                    },
                    a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
                    a03: 0 as *u8,
                    a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
                    a07: 0 as *u8,
                    a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
                    a11: 0 as *u8,
                    a12: 0 as *u8, a13: 0 as *u8, a14: 0 as *u8,
                    a15: 0 as *u8,
                    a16: 0 as *u8, a17: 0 as *u8, a18: 0 as *u8,
                    a19: 0 as *u8,
                    a20: 0 as *u8, a21: 0 as *u8,
                    a22: uv_tcp_t_32bit_unix_riders { a29: 0 as *u8 },
                }
            }
            #[cfg(target_arch="x86")]
            #[cfg(target_arch="arm")]
            #[cfg(target_arch="mips")]
            pub fn gen_stub_arch() -> uv_tcp_t {
                uv_tcp_t {
                    fields: uv_handle_fields {
                        loop_handle: ptr::null(), type_: 0u32,
                        close_cb: ptr::null(),
                        data: ptr::null(),
                    },
                    a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
                    a03: 0 as *u8,
                    a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
                    a07: 0 as *u8,
                    a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
                    a11: 0 as *u8,
                    a12: 0 as *u8, a13: 0 as *u8, a14: 0 as *u8,
                    a15: 0 as *u8,
                    a16: 0 as *u8, a17: 0 as *u8, a18: 0 as *u8,
                    a19: 0 as *u8,
                    a20: 0 as *u8, a21: 0 as *u8,
                    a22: uv_tcp_t_32bit_unix_riders {
                        a29: 0 as *u8, a30: 0 as *u8, a31: 0 as *u8,
                    },
                }
            }
        }
        #[cfg(windows)]
        pub fn gen_stub_os() -> uv_tcp_t {
            uv_tcp_t {
                fields: uv_handle_fields {
                    loop_handle: ptr::null(), type_: 0u32,
                    close_cb: ptr::null(),
                    data: ptr::null(),
                },
                a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
                a03: 0 as *u8,
                a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
                a07: 0 as *u8,
                a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
                a11: 0 as *u8,
                a12: 0 as *u8, a13: 0 as *u8, a14: 0 as *u8,
                a15: 0 as *u8,
                a16: 0 as *u8, a17: 0 as *u8, a18: 0 as *u8,
                a19: 0 as *u8,
                a20: 0 as *u8, a21: 0 as *u8, a22: 0 as *u8,
                a23: 0 as *u8,
                a24: 0 as *u8, a25: 0 as *u8,
            }
        }
        #[cfg(target_os = "macos")]
        pub fn gen_stub_os() -> uv_tcp_t {
            use super::uv_tcp_t_osx_riders;

            return gen_stub_arch();

            #[cfg(target_arch = "x86_64")]
            fn gen_stub_arch() -> uv_tcp_t {
                uv_tcp_t {
                    fields: uv_handle_fields {
                        loop_handle: ptr::null(), type_: 0u32,
                        close_cb: ptr::null(),
                        data: ptr::null(),
                    },
                    a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
                    a03: 0 as *u8,
                    a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
                    a07: 0 as *u8,
                    a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
                    a11: 0 as *u8,
                    a12: 0 as *u8, a13: 0 as *u8, a14: 0 as *u8,
                    a15: 0 as *u8,
                    a16: 0 as *u8, a17: 0 as *u8, a18: 0 as *u8,
                    a19: 0 as *u8,
                    a20: 0 as *u8, a21: 0 as *u8, a22: 0 as *u8,
                    a23: uv_tcp_t_osx_riders {
                        a23: 0 as *u8,
                    }
                }
            }

            #[cfg(target_arch = "x86")]
            #[cfg(target_arch = "arm")]
            fn gen_stub_arch() -> uv_tcp_t {
                uv_tcp_t {
                    fields: uv_handle_fields {
                        loop_handle: ptr::null(), type_: 0u32,
                        close_cb: ptr::null(),
                        data: ptr::null(),
                    },
                    a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
                    a03: 0 as *u8,
                    a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
                    a07: 0 as *u8,
                    a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
                    a11: 0 as *u8,
                    a12: 0 as *u8, a13: 0 as *u8, a14: 0 as *u8,
                    a15: 0 as *u8,
                    a16: 0 as *u8, a17: 0 as *u8, a18: 0 as *u8,
                    a19: 0 as *u8,
                    a20: 0 as *u8, a21: 0 as *u8, a22: 0 as *u8,
                    a23: uv_tcp_t_osx_riders {
                        a23: 0 as *u8,
                        a24: 0 as *u8, a25: 0 as *u8,
                    }
                }
            }
        }
    }
    #[cfg(unix)]
    pub fn gen_stub_uv_connect_t() -> uv_connect_t {
        uv_connect_t {
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
            a07: 0 as *u8
        }
    }
    #[cfg(windows)]
    pub fn gen_stub_uv_connect_t() -> uv_connect_t {
        uv_connect_t {
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
            a07: 0 as *u8,
            a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
        }
    }
    #[cfg(unix)]
    pub fn gen_stub_uv_async_t() -> uv_async_t {
        return gen_stub_arch();
        #[cfg(target_arch = "x86_64")]
        pub fn gen_stub_arch() -> uv_async_t {
            uv_async_t {
                fields: uv_handle_fields {
                    loop_handle: ptr::null(), type_: 0u32,
                    close_cb: ptr::null(),
                    data: ptr::null(),
                },
                a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
                a03: 0 as *u8,
                a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
                a07: uv_async_t_32bit_unix_riders { a10: 0 as *u8 },
            }
        }
        #[cfg(target_arch = "x86")]
        #[cfg(target_arch="arm")]
        #[cfg(target_arch="mips")]
        pub fn gen_stub_arch() -> uv_async_t {
            uv_async_t {
                fields: uv_handle_fields {
                    loop_handle: ptr::null(), type_: 0u32,
                    close_cb: ptr::null(),
                    data: ptr::null(),
                },
                a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
                a03: 0 as *u8,
                a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
                a07: uv_async_t_32bit_unix_riders {
                    a10: 0 as *u8,
                }
            }
        }
    }
    #[cfg(windows)]
    pub fn gen_stub_uv_async_t() -> uv_async_t {
        uv_async_t {
            fields: uv_handle_fields {
                loop_handle: ptr::null(), type_: 0u32,
                close_cb: ptr::null(),
                data: ptr::null(),
            },
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
            a07: 0 as *u8,
            a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
            a11: 0 as *u8,
            a12: 0 as *u8,
        }
    }
    #[cfg(unix)]
    pub fn gen_stub_uv_timer_t() -> uv_timer_t {
        return gen_stub_arch();
        #[cfg(target_arch = "x86_64")]
        pub fn gen_stub_arch() -> uv_timer_t {
            uv_timer_t {
                fields: uv_handle_fields {
                    loop_handle: ptr::null(), type_: 0u32,
                    close_cb: ptr::null(),
                    data: ptr::null(),
                },
                a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
                a03: 0 as *u8,
                a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
                a07: 0 as *u8,
                a08: 0 as *u8, a09: 0 as *u8,
                a11: uv_timer_t_32bit_unix_riders {
                    a10: 0 as *u8
                },
            }
        }
        #[cfg(target_arch = "x86")]
        #[cfg(target_arch="arm")]
        #[cfg(target_arch="mips")]
        pub fn gen_stub_arch() -> uv_timer_t {
            uv_timer_t {
                fields: uv_handle_fields {
                    loop_handle: ptr::null(), type_: 0u32,
                    close_cb: ptr::null(),
                    data: ptr::null(),
                },
                a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
                a03: 0 as *u8,
                a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
                a07: 0 as *u8,
                a08: 0 as *u8, a09: 0 as *u8,
                a11: uv_timer_t_32bit_unix_riders {
                    a10: 0 as *u8, a11: 0 as *u8,
                    a12: 0 as *u8,
                },
            }
        }
    }
    #[cfg(windows)]
    pub fn gen_stub_uv_timer_t() -> uv_timer_t {
        uv_timer_t {
            fields: uv_handle_fields {
                loop_handle: ptr::null(), type_: 0u32,
                close_cb: ptr::null(),
                data: ptr::null(),
            },
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
            a07: 0 as *u8,
            a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
            a11: 0 as *u8,
        }
    }
    #[cfg(unix)]
    pub fn gen_stub_uv_write_t() -> uv_write_t {
        return gen_stub_arch();
        #[cfg(target_arch="x86_64")]
        pub fn gen_stub_arch() -> uv_write_t {
            uv_write_t {
                fields: uv_handle_fields {
                    loop_handle: ptr::null(), type_: 0u32,
                    close_cb: ptr::null(),
                    data: ptr::null(),
                },
                a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
                a03: 0 as *u8,
                a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
                a07: 0 as *u8,
                a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
                a11: 0 as *u8,
                a12: 0 as *u8,
                a14: uv_write_t_32bit_unix_riders { a13: 0 as *u8,
                                                   a14: 0 as *u8,
                                                   a15: 0 as *u8},
            }
        }
        #[cfg(target_arch="x86")]
        #[cfg(target_arch="arm")]
        #[cfg(target_arch="mips")]
        pub fn gen_stub_arch() -> uv_write_t {
            uv_write_t {
                fields: uv_handle_fields {
                    loop_handle: ptr::null(), type_: 0u32,
                    close_cb: ptr::null(),
                    data: ptr::null(),
                },
                a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
                a03: 0 as *u8,
                a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
                a07: 0 as *u8,
                a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
                a11: 0 as *u8,
                a12: 0 as *u8,
                a14: uv_write_t_32bit_unix_riders {
                    a13: 0 as *u8,
                    a14: 0 as *u8,
                    a15: 0 as *u8,
                    a16: 0 as *u8,
                }
            }
        }
    }
    #[cfg(windows)]
    pub fn gen_stub_uv_write_t() -> uv_write_t {
        uv_write_t {
            fields: uv_handle_fields {
                loop_handle: ptr::null(), type_: 0u32,
                close_cb: ptr::null(),
                data: ptr::null(),
            },
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
            a07: 0 as *u8,
            a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
            a11: 0 as *u8,
            a12: 0 as *u8
        }
    }
    pub fn gen_stub_uv_getaddrinfo_t() -> uv_getaddrinfo_t {
        uv_getaddrinfo_t {
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8, a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8, a07: 0 as *u8,
            a08: 0 as *u8, a09: 0 as *u8,
            a10: 1 as *u8, a11: 1 as *u8, a12: 1 as *u8, a13: 1 as *u8,
            a14: 1 as *u8, a15: 1 as *u8
        }
    }
}

#[nolink]
extern {
    // libuv public API
    unsafe fn rust_uv_loop_new() -> *libc::c_void;
    unsafe fn rust_uv_loop_delete(lp: *libc::c_void);
    unsafe fn rust_uv_run(loop_handle: *libc::c_void);
    unsafe fn rust_uv_close(handle: *libc::c_void, cb: *u8);
    unsafe fn rust_uv_walk(loop_handle: *libc::c_void, cb: *u8,
                           arg: *libc::c_void);

    unsafe fn rust_uv_idle_new() -> *uv_idle_t;
    unsafe fn rust_uv_idle_delete(handle: *uv_idle_t);
    unsafe fn rust_uv_idle_init(loop_handle: *uv_loop_t,
                                handle: *uv_idle_t) -> libc::c_int;
    unsafe fn rust_uv_idle_start(handle: *uv_idle_t,
                                 cb: uv_idle_cb) -> libc::c_int;
    unsafe fn rust_uv_idle_stop(handle: *uv_idle_t) -> libc::c_int;

    unsafe fn rust_uv_async_send(handle: *uv_async_t);
    unsafe fn rust_uv_async_init(loop_handle: *libc::c_void,
                          async_handle: *uv_async_t,
                          cb: *u8) -> libc::c_int;
    unsafe fn rust_uv_tcp_init(
        loop_handle: *libc::c_void,
        handle_ptr: *uv_tcp_t) -> libc::c_int;
    // FIXME ref #2604 .. ?
    unsafe fn rust_uv_buf_init(out_buf: *uv_buf_t, base: *u8,
                        len: libc::size_t);
    unsafe fn rust_uv_last_error(loop_handle: *libc::c_void) -> uv_err_t;
    // FIXME ref #2064
    unsafe fn rust_uv_strerror(err: *uv_err_t) -> *libc::c_char;
    // FIXME ref #2064
    unsafe fn rust_uv_err_name(err: *uv_err_t) -> *libc::c_char;
    unsafe fn rust_uv_ip4_addr(ip: *u8, port: libc::c_int)
        -> sockaddr_in;
    unsafe fn rust_uv_ip6_addr(ip: *u8, port: libc::c_int)
        -> sockaddr_in6;
    unsafe fn rust_uv_ip4_name(src: *sockaddr_in,
                               dst: *u8,
                               size: libc::size_t)
                            -> libc::c_int;
    unsafe fn rust_uv_ip6_name(src: *sockaddr_in6,
                               dst: *u8,
                               size: libc::size_t)
                            -> libc::c_int;
    unsafe fn rust_uv_ip4_port(src: *sockaddr_in) -> libc::c_uint;
    unsafe fn rust_uv_ip6_port(src: *sockaddr_in6) -> libc::c_uint;
    // FIXME ref #2064
    unsafe fn rust_uv_tcp_connect(connect_ptr: *uv_connect_t,
                                  tcp_handle_ptr: *uv_tcp_t,
                                  after_cb: *u8,
                                  addr: *sockaddr_in)
                                  -> libc::c_int;
    // FIXME ref #2064
    unsafe fn rust_uv_tcp_bind(tcp_server: *uv_tcp_t,
                               addr: *sockaddr_in) -> libc::c_int;
    // FIXME ref #2064
    unsafe fn rust_uv_tcp_connect6(connect_ptr: *uv_connect_t,
                                   tcp_handle_ptr: *uv_tcp_t,
                                   after_cb: *u8,
                                   addr: *sockaddr_in6) -> libc::c_int;
    // FIXME ref #2064
    unsafe fn rust_uv_tcp_bind6(tcp_server: *uv_tcp_t,
                                addr: *sockaddr_in6) -> libc::c_int;
    unsafe fn rust_uv_tcp_getpeername(tcp_handle_ptr: *uv_tcp_t,
                                      name: *sockaddr_in) -> libc::c_int;
    unsafe fn rust_uv_tcp_getpeername6(tcp_handle_ptr: *uv_tcp_t,
                                       name: *sockaddr_in6) ->libc::c_int;
    unsafe fn rust_uv_listen(stream: *libc::c_void,
                             backlog: libc::c_int,
                             cb: *u8) -> libc::c_int;
    unsafe fn rust_uv_accept(server: *libc::c_void, client: *libc::c_void)
                          -> libc::c_int;
    unsafe fn rust_uv_write(req: *libc::c_void,
                            stream: *libc::c_void,
                            buf_in: *uv_buf_t,
                            buf_cnt: libc::c_int,
                            cb: *u8)
        -> libc::c_int;
    unsafe fn rust_uv_read_start(stream: *libc::c_void,
                                 on_alloc: *u8,
                                 on_read: *u8)
        -> libc::c_int;
    unsafe fn rust_uv_read_stop(stream: *libc::c_void) -> libc::c_int;
    unsafe fn rust_uv_timer_init(loop_handle: *libc::c_void,
                                 timer_handle: *uv_timer_t)
        -> libc::c_int;
    unsafe fn rust_uv_timer_start(
        timer_handle: *uv_timer_t,
        cb: *u8,
        timeout: libc::uint64_t,
        repeat: libc::uint64_t) -> libc::c_int;
    unsafe fn rust_uv_timer_stop(handle: *uv_timer_t) -> libc::c_int;

    unsafe fn rust_uv_getaddrinfo(loop_ptr: *libc::c_void,
                                  handle: *uv_getaddrinfo_t,
                                  cb: *u8,
                                  node_name_ptr: *u8,
                                  service_name_ptr: *u8,
                                  // should probably only pass ptr::null()
                                  hints: *addrinfo)
        -> libc::c_int;
    unsafe fn rust_uv_freeaddrinfo(res: *addrinfo);

    // data accessors/helpers for rust-mapped uv structs
    unsafe fn rust_uv_helper_get_INADDR_NONE() -> u32;
    unsafe fn rust_uv_is_ipv4_addrinfo(input: *addrinfo) -> bool;
    unsafe fn rust_uv_is_ipv6_addrinfo(input: *addrinfo) -> bool;
    unsafe fn rust_uv_get_next_addrinfo(input: *addrinfo) -> *addrinfo;
    unsafe fn rust_uv_addrinfo_as_sockaddr_in(input: *addrinfo)
        -> *sockaddr_in;
    unsafe fn rust_uv_addrinfo_as_sockaddr_in6(input: *addrinfo)
        -> *sockaddr_in6;
    unsafe fn rust_uv_malloc_buf_base_of(sug_size: libc::size_t) -> *u8;
    unsafe fn rust_uv_free_base_of_buf(buf: uv_buf_t);
    unsafe fn rust_uv_get_stream_handle_from_connect_req(
        connect_req: *uv_connect_t)
        -> *uv_stream_t;
    unsafe fn rust_uv_get_stream_handle_from_write_req(
        write_req: *uv_write_t)
        -> *uv_stream_t;
    unsafe fn rust_uv_get_loop_for_uv_handle(handle: *libc::c_void)
        -> *libc::c_void;
    unsafe fn rust_uv_get_data_for_uv_loop(loop_ptr: *libc::c_void)
        -> *libc::c_void;
    unsafe fn rust_uv_set_data_for_uv_loop(loop_ptr: *libc::c_void,
                                           data: *libc::c_void);
    unsafe fn rust_uv_get_data_for_uv_handle(handle: *libc::c_void)
        -> *libc::c_void;
    unsafe fn rust_uv_set_data_for_uv_handle(handle: *libc::c_void,
                                             data: *libc::c_void);
    unsafe fn rust_uv_get_data_for_req(req: *libc::c_void)
        -> *libc::c_void;
    unsafe fn rust_uv_set_data_for_req(req: *libc::c_void,
                                       data: *libc::c_void);
    unsafe fn rust_uv_get_base_from_buf(buf: uv_buf_t) -> *u8;
    unsafe fn rust_uv_get_len_from_buf(buf: uv_buf_t) -> libc::size_t;

    // sizeof testing helpers
    unsafe fn rust_uv_helper_uv_tcp_t_size() -> libc::c_uint;
    unsafe fn rust_uv_helper_uv_connect_t_size() -> libc::c_uint;
    unsafe fn rust_uv_helper_uv_buf_t_size() -> libc::c_uint;
    unsafe fn rust_uv_helper_uv_write_t_size() -> libc::c_uint;
    unsafe fn rust_uv_helper_uv_err_t_size() -> libc::c_uint;
    unsafe fn rust_uv_helper_sockaddr_in_size() -> libc::c_uint;
    unsafe fn rust_uv_helper_sockaddr_in6_size() -> libc::c_uint;
    unsafe fn rust_uv_helper_uv_async_t_size() -> libc::c_uint;
    unsafe fn rust_uv_helper_uv_timer_t_size() -> libc::c_uint;
    unsafe fn rust_uv_helper_uv_getaddrinfo_t_size() -> libc::c_uint;
    unsafe fn rust_uv_helper_addrinfo_size() -> libc::c_uint;
    unsafe fn rust_uv_helper_addr_in_size() -> libc::c_uint;
}

pub unsafe fn loop_new() -> *libc::c_void {
    return rust_uv_loop_new();
}

pub unsafe fn loop_delete(loop_handle: *libc::c_void) {
    rust_uv_loop_delete(loop_handle);
}

pub unsafe fn run(loop_handle: *libc::c_void) {
    rust_uv_run(loop_handle);
}

pub unsafe fn close<T>(handle: *T, cb: *u8) {
    rust_uv_close(handle as *libc::c_void, cb);
}

pub unsafe fn walk(loop_handle: *libc::c_void, cb: *u8, arg: *libc::c_void) {
    rust_uv_walk(loop_handle, cb, arg);
}

pub unsafe fn idle_new() -> *uv_idle_t {
    rust_uv_idle_new()
}

pub unsafe fn idle_delete(handle: *uv_idle_t) {
    rust_uv_idle_delete(handle)
}

pub unsafe fn idle_init(loop_handle: *uv_loop_t,
                        handle: *uv_idle_t) -> libc::c_int {
    rust_uv_idle_init(loop_handle, handle)
}

pub unsafe fn idle_start(handle: *uv_idle_t, cb: uv_idle_cb) -> libc::c_int {
    rust_uv_idle_start(handle, cb)
}

pub unsafe fn idle_stop(handle: *uv_idle_t) -> libc::c_int {
    rust_uv_idle_stop(handle)
}

pub unsafe fn tcp_init(loop_handle: *libc::c_void, handle: *uv_tcp_t)
    -> libc::c_int {
    return rust_uv_tcp_init(loop_handle, handle);
}
// FIXME ref #2064
pub unsafe fn tcp_connect(connect_ptr: *uv_connect_t,
                      tcp_handle_ptr: *uv_tcp_t,
                      addr_ptr: *sockaddr_in,
                      after_connect_cb: *u8)
-> libc::c_int {
    return rust_uv_tcp_connect(connect_ptr, tcp_handle_ptr,
                                    after_connect_cb, addr_ptr);
}
// FIXME ref #2064
pub unsafe fn tcp_connect6(connect_ptr: *uv_connect_t,
                      tcp_handle_ptr: *uv_tcp_t,
                      addr_ptr: *sockaddr_in6,
                      after_connect_cb: *u8)
-> libc::c_int {
    return rust_uv_tcp_connect6(connect_ptr, tcp_handle_ptr,
                                    after_connect_cb, addr_ptr);
}
// FIXME ref #2064
pub unsafe fn tcp_bind(tcp_server_ptr: *uv_tcp_t,
                   addr_ptr: *sockaddr_in) -> libc::c_int {
    return rust_uv_tcp_bind(tcp_server_ptr,
                                 addr_ptr);
}
// FIXME ref #2064
pub unsafe fn tcp_bind6(tcp_server_ptr: *uv_tcp_t,
                   addr_ptr: *sockaddr_in6) -> libc::c_int {
    return rust_uv_tcp_bind6(tcp_server_ptr,
                                 addr_ptr);
}

pub unsafe fn tcp_getpeername(tcp_handle_ptr: *uv_tcp_t,
                              name: *sockaddr_in) -> libc::c_int {
    return rust_uv_tcp_getpeername(tcp_handle_ptr, name);
}

pub unsafe fn tcp_getpeername6(tcp_handle_ptr: *uv_tcp_t,
                               name: *sockaddr_in6) ->libc::c_int {
    return rust_uv_tcp_getpeername6(tcp_handle_ptr, name);
}

pub unsafe fn listen<T>(stream: *T, backlog: libc::c_int,
                 cb: *u8) -> libc::c_int {
    return rust_uv_listen(stream as *libc::c_void, backlog, cb);
}

pub unsafe fn accept(server: *libc::c_void, client: *libc::c_void)
    -> libc::c_int {
    return rust_uv_accept(server as *libc::c_void,
                               client as *libc::c_void);
}

pub unsafe fn write<T>(req: *uv_write_t, stream: *T,
         buf_in: *~[uv_buf_t], cb: *u8) -> libc::c_int {
    let buf_ptr = vec::raw::to_ptr(*buf_in);
    let buf_cnt = vec::len(*buf_in) as i32;
    return rust_uv_write(req as *libc::c_void,
                              stream as *libc::c_void,
                              buf_ptr, buf_cnt, cb);
}
pub unsafe fn read_start(stream: *uv_stream_t, on_alloc: *u8,
                     on_read: *u8) -> libc::c_int {
    return rust_uv_read_start(stream as *libc::c_void,
                                   on_alloc, on_read);
}

pub unsafe fn read_stop(stream: *uv_stream_t) -> libc::c_int {
    return rust_uv_read_stop(stream as *libc::c_void);
}

pub unsafe fn last_error(loop_handle: *libc::c_void) -> uv_err_t {
    return rust_uv_last_error(loop_handle);
}

pub unsafe fn strerror(err: *uv_err_t) -> *libc::c_char {
    return rust_uv_strerror(err);
}
pub unsafe fn err_name(err: *uv_err_t) -> *libc::c_char {
    return rust_uv_err_name(err);
}

pub unsafe fn async_init(loop_handle: *libc::c_void,
                     async_handle: *uv_async_t,
                     cb: *u8) -> libc::c_int {
    return rust_uv_async_init(loop_handle,
                                   async_handle,
                                   cb);
}

pub unsafe fn async_send(async_handle: *uv_async_t) {
    return rust_uv_async_send(async_handle);
}
pub unsafe fn buf_init(input: *u8, len: uint) -> uv_buf_t {
    let out_buf = uv_buf_t { base: ptr::null(), len: 0 as libc::size_t };
    let out_buf_ptr: *uv_buf_t = &out_buf;
    rust_uv_buf_init(out_buf_ptr, input, len as size_t);
    return out_buf;
}
pub unsafe fn ip4_addr(ip: &str, port: int) -> sockaddr_in {
    do str::as_c_str(ip) |ip_buf| {
        rust_uv_ip4_addr(ip_buf as *u8,
                                 port as libc::c_int)
    }
}
pub unsafe fn ip6_addr(ip: &str, port: int) -> sockaddr_in6 {
    do str::as_c_str(ip) |ip_buf| {
        rust_uv_ip6_addr(ip_buf as *u8,
                                 port as libc::c_int)
    }
}
pub unsafe fn ip4_name(src: &sockaddr_in) -> ~str {
    // ipv4 addr max size: 15 + 1 trailing null byte
    let dst: ~[u8] = ~[0u8,0u8,0u8,0u8,0u8,0u8,0u8,0u8,
                     0u8,0u8,0u8,0u8,0u8,0u8,0u8,0u8];
    do vec::as_imm_buf(dst) |dst_buf, size| {
        rust_uv_ip4_name(to_unsafe_ptr(src),
                                 dst_buf, size as libc::size_t);
        // seems that checking the result of uv_ip4_name
        // doesn't work too well..
        // you're stuck looking at the value of dst_buf
        // to see if it is the string representation of
        // INADDR_NONE (0xffffffff or 255.255.255.255 on
        // many platforms)
        str::raw::from_buf(dst_buf)
    }
}
pub unsafe fn ip6_name(src: &sockaddr_in6) -> ~str {
    // ipv6 addr max size: 45 + 1 trailing null byte
    let dst: ~[u8] = ~[0u8,0u8,0u8,0u8,0u8,0u8,0u8,0u8,
                       0u8,0u8,0u8,0u8,0u8,0u8,0u8,0u8,
                       0u8,0u8,0u8,0u8,0u8,0u8,0u8,0u8,
                       0u8,0u8,0u8,0u8,0u8,0u8,0u8,0u8,
                       0u8,0u8,0u8,0u8,0u8,0u8,0u8,0u8,
                       0u8,0u8,0u8,0u8,0u8,0u8];
    do vec::as_imm_buf(dst) |dst_buf, size| {
        let src_unsafe_ptr = to_unsafe_ptr(src);
        let result = rust_uv_ip6_name(src_unsafe_ptr,
                                              dst_buf, size as libc::size_t);
        match result {
          0i32 => str::raw::from_buf(dst_buf),
          _ => ~""
        }
    }
}
pub unsafe fn ip4_port(src: &sockaddr_in) -> uint {
    rust_uv_ip4_port(to_unsafe_ptr(src)) as uint
}
pub unsafe fn ip6_port(src: &sockaddr_in6) -> uint {
    rust_uv_ip6_port(to_unsafe_ptr(src)) as uint
}

pub unsafe fn timer_init(loop_ptr: *libc::c_void,
                     timer_ptr: *uv_timer_t) -> libc::c_int {
    return rust_uv_timer_init(loop_ptr, timer_ptr);
}
pub unsafe fn timer_start(timer_ptr: *uv_timer_t, cb: *u8, timeout: uint,
                      repeat: uint) -> libc::c_int {
    return rust_uv_timer_start(timer_ptr, cb, timeout as libc::uint64_t,
                               repeat as libc::uint64_t);
}
pub unsafe fn timer_stop(timer_ptr: *uv_timer_t) -> libc::c_int {
    return rust_uv_timer_stop(timer_ptr);
}
pub unsafe fn getaddrinfo(loop_ptr: *libc::c_void,
                           handle: *uv_getaddrinfo_t,
                           cb: *u8,
                           node_name_ptr: *u8,
                           service_name_ptr: *u8,
                           hints: *addrinfo) -> libc::c_int {
    rust_uv_getaddrinfo(loop_ptr,
                           handle,
                           cb,
                           node_name_ptr,
                           service_name_ptr,
                           hints)
}
pub unsafe fn freeaddrinfo(res: *addrinfo) {
    rust_uv_freeaddrinfo(res);
}

// libuv struct initializers
pub fn tcp_t() -> uv_tcp_t {
    return uv_ll_struct_stubgen::gen_stub_uv_tcp_t();
}
pub fn connect_t() -> uv_connect_t {
    return uv_ll_struct_stubgen::gen_stub_uv_connect_t();
}
pub fn write_t() -> uv_write_t {
    return uv_ll_struct_stubgen::gen_stub_uv_write_t();
}
pub fn async_t() -> uv_async_t {
    return uv_ll_struct_stubgen::gen_stub_uv_async_t();
}
pub fn timer_t() -> uv_timer_t {
    return uv_ll_struct_stubgen::gen_stub_uv_timer_t();
}
pub fn getaddrinfo_t() -> uv_getaddrinfo_t {
    return uv_ll_struct_stubgen::gen_stub_uv_getaddrinfo_t();
}

// data access helpers
pub unsafe fn get_loop_for_uv_handle<T>(handle: *T)
    -> *libc::c_void {
    return rust_uv_get_loop_for_uv_handle(handle as *libc::c_void);
}
pub unsafe fn get_stream_handle_from_connect_req(connect: *uv_connect_t)
    -> *uv_stream_t {
    return rust_uv_get_stream_handle_from_connect_req(
        connect);
}
pub unsafe fn get_stream_handle_from_write_req(
    write_req: *uv_write_t)
    -> *uv_stream_t {
    return rust_uv_get_stream_handle_from_write_req(
        write_req);
}
pub unsafe fn get_data_for_uv_loop(loop_ptr: *libc::c_void) -> *libc::c_void {
    rust_uv_get_data_for_uv_loop(loop_ptr)
}
pub unsafe fn set_data_for_uv_loop(loop_ptr: *libc::c_void,
                                   data: *libc::c_void) {
    rust_uv_set_data_for_uv_loop(loop_ptr, data);
}
pub unsafe fn get_data_for_uv_handle<T>(handle: *T) -> *libc::c_void {
    return rust_uv_get_data_for_uv_handle(handle as *libc::c_void);
}
pub unsafe fn set_data_for_uv_handle<T, U>(handle: *T, data: *U) {
    rust_uv_set_data_for_uv_handle(handle as *libc::c_void,
                                           data as *libc::c_void);
}
pub unsafe fn get_data_for_req<T>(req: *T) -> *libc::c_void {
    return rust_uv_get_data_for_req(req as *libc::c_void);
}
pub unsafe fn set_data_for_req<T, U>(req: *T,
                    data: *U) {
    rust_uv_set_data_for_req(req as *libc::c_void,
                                     data as *libc::c_void);
}
pub unsafe fn get_base_from_buf(buf: uv_buf_t) -> *u8 {
    return rust_uv_get_base_from_buf(buf);
}
pub unsafe fn get_len_from_buf(buf: uv_buf_t) -> libc::size_t {
    return rust_uv_get_len_from_buf(buf);
}
pub unsafe fn malloc_buf_base_of(suggested_size: libc::size_t)
    -> *u8 {
    return rust_uv_malloc_buf_base_of(suggested_size);
}
pub unsafe fn free_base_of_buf(buf: uv_buf_t) {
    rust_uv_free_base_of_buf(buf);
}

pub unsafe fn get_last_err_info(uv_loop: *libc::c_void) -> ~str {
    let err = last_error(uv_loop);
    let err_ptr: *uv_err_t = &err;
    let err_name = str::raw::from_c_str(err_name(err_ptr));
    let err_msg = str::raw::from_c_str(strerror(err_ptr));
    return fmt!("LIBUV ERROR: name: %s msg: %s",
                    err_name, err_msg);
}

pub unsafe fn get_last_err_data(uv_loop: *libc::c_void) -> uv_err_data {
    let err = last_error(uv_loop);
    let err_ptr: *uv_err_t = &err;
    let err_name = str::raw::from_c_str(err_name(err_ptr));
    let err_msg = str::raw::from_c_str(strerror(err_ptr));
    uv_err_data { err_name: err_name, err_msg: err_msg }
}

pub struct uv_err_data {
    err_name: ~str,
    err_msg: ~str,
}

pub unsafe fn is_ipv4_addrinfo(input: *addrinfo) -> bool {
    rust_uv_is_ipv4_addrinfo(input)
}
pub unsafe fn is_ipv6_addrinfo(input: *addrinfo) -> bool {
    rust_uv_is_ipv6_addrinfo(input)
}
pub unsafe fn get_INADDR_NONE() -> u32 {
    rust_uv_helper_get_INADDR_NONE()
}
pub unsafe fn get_next_addrinfo(input: *addrinfo) -> *addrinfo {
    rust_uv_get_next_addrinfo(input)
}
pub unsafe fn addrinfo_as_sockaddr_in(input: *addrinfo) -> *sockaddr_in {
    rust_uv_addrinfo_as_sockaddr_in(input)
}
pub unsafe fn addrinfo_as_sockaddr_in6(input: *addrinfo) -> *sockaddr_in6 {
    rust_uv_addrinfo_as_sockaddr_in6(input)
}

#[cfg(test)]
mod test {
    use core::comm::{SharedChan, stream, GenericChan, GenericPort};
    use super::*;

    enum tcp_read_data {
        tcp_read_eof,
        tcp_read_more(~[u8]),
        tcp_read_error
    }

    struct request_wrapper {
        write_req: *uv_write_t,
        req_buf: *~[uv_buf_t],
        read_chan: SharedChan<~str>,
    }

    extern fn after_close_cb(handle: *libc::c_void) {
        debug!("after uv_close! handle ptr: %?",
                        handle);
    }

    extern fn on_alloc_cb(handle: *libc::c_void,
                         suggested_size: libc::size_t)
        -> uv_buf_t {
        unsafe {
            debug!(~"on_alloc_cb!");
            let char_ptr = malloc_buf_base_of(suggested_size);
            debug!("on_alloc_cb h: %? char_ptr: %u sugsize: %u",
                             handle,
                             char_ptr as uint,
                             suggested_size as uint);
            return buf_init(char_ptr, suggested_size as uint);
        }
    }

    extern fn on_read_cb(stream: *uv_stream_t,
                        nread: libc::ssize_t,
                        buf: uv_buf_t) {
        unsafe {
            let nread = nread as int;
            debug!("CLIENT entering on_read_cb nred: %d",
                            nread);
            if (nread > 0) {
                // we have data
                debug!("CLIENT read: data! nread: %d", nread);
                read_stop(stream);
                let client_data =
                    get_data_for_uv_handle(stream as *libc::c_void)
                      as *request_wrapper;
                let buf_base = get_base_from_buf(buf);
                let bytes = vec::from_buf(buf_base, nread as uint);
                let read_chan = (*client_data).read_chan.clone();
                let msg_from_server = str::from_bytes(bytes);
                read_chan.send(msg_from_server);
                close(stream as *libc::c_void, after_close_cb)
            }
            else if (nread == -1) {
                // err .. possibly EOF
                debug!(~"read: eof!");
            }
            else {
                // nread == 0 .. do nothing, just free buf as below
                debug!(~"read: do nothing!");
            }
            // when we're done
            free_base_of_buf(buf);
            debug!(~"CLIENT exiting on_read_cb");
        }
    }

    extern fn on_write_complete_cb(write_req: *uv_write_t,
                                  status: libc::c_int) {
        unsafe {
            debug!(
                "CLIENT beginning on_write_complete_cb status: %d",
                     status as int);
            let stream = get_stream_handle_from_write_req(write_req);
            debug!(
                "CLIENT on_write_complete_cb: tcp:%d write_handle:%d",
                stream as int, write_req as int);
            let result = read_start(stream, on_alloc_cb, on_read_cb);
            debug!("CLIENT ending on_write_complete_cb .. status: %d",
                     result as int);
        }
    }

    extern fn on_connect_cb(connect_req_ptr: *uv_connect_t,
                                 status: libc::c_int) {
        unsafe {
            debug!("beginning on_connect_cb .. status: %d",
                             status as int);
            let stream =
                get_stream_handle_from_connect_req(connect_req_ptr);
            if (status == 0i32) {
                debug!(~"on_connect_cb: in status=0 if..");
                let client_data = get_data_for_req(
                    connect_req_ptr as *libc::c_void)
                    as *request_wrapper;
                let write_handle = (*client_data).write_req;
                debug!("on_connect_cb: tcp: %d write_hdl: %d",
                                stream as int, write_handle as int);
                let write_result = write(write_handle,
                                  stream as *libc::c_void,
                                  (*client_data).req_buf,
                                  on_write_complete_cb);
                debug!("on_connect_cb: write() status: %d",
                                 write_result as int);
            }
            else {
                let test_loop = get_loop_for_uv_handle(
                    stream as *libc::c_void);
                let err_msg = get_last_err_info(test_loop);
                debug!(err_msg);
                assert!(false);
            }
            debug!(~"finishing on_connect_cb");
        }
    }

    fn impl_uv_tcp_request(ip: &str, port: int, req_str: &str,
                          client_chan: SharedChan<~str>) {
        unsafe {
            let test_loop = loop_new();
            let tcp_handle = tcp_t();
            let tcp_handle_ptr: *uv_tcp_t = &tcp_handle;
            let connect_handle = connect_t();
            let connect_req_ptr: *uv_connect_t = &connect_handle;

            // this is the persistent payload of data that we
            // need to pass around to get this example to work.
            // In C, this would be a malloc'd or stack-allocated
            // struct that we'd cast to a void* and store as the
            // data field in our uv_connect_t struct
            let req_str_bytes = str::to_bytes(req_str);
            let req_msg_ptr: *u8 = vec::raw::to_ptr(req_str_bytes);
            debug!("req_msg ptr: %u", req_msg_ptr as uint);
            let req_msg = ~[
                buf_init(req_msg_ptr, req_str_bytes.len())
            ];
            // this is the enclosing record, we'll pass a ptr to
            // this to C..
            let write_handle = write_t();
            let write_handle_ptr: *uv_write_t = &write_handle;
            debug!("tcp req: tcp stream: %d write_handle: %d",
                             tcp_handle_ptr as int,
                             write_handle_ptr as int);
            let client_data = request_wrapper {
                write_req: write_handle_ptr,
                req_buf: &req_msg,
                read_chan: client_chan
            };

            let tcp_init_result = tcp_init(test_loop as *libc::c_void,
                                           tcp_handle_ptr);
            if (tcp_init_result == 0) {
                debug!(~"successful tcp_init_result");

                debug!(~"building addr...");
                let addr = ip4_addr(ip, port);
                // FIXME ref #2064
                let addr_ptr: *sockaddr_in = &addr;
                debug!("after build addr in rust. port: %u",
                       addr.sin_port as uint);

                // this should set up the connection request..
                debug!("b4 call tcp_connect connect cb: %u ",
                       on_connect_cb as uint);
                let tcp_connect_result = tcp_connect(connect_req_ptr,
                                                     tcp_handle_ptr,
                                                     addr_ptr,
                                                     on_connect_cb);
                if (tcp_connect_result == 0) {
                    // not set the data on the connect_req
                    // until its initialized
                    set_data_for_req(connect_req_ptr as *libc::c_void,
                                     &client_data);
                    set_data_for_uv_handle(tcp_handle_ptr as *libc::c_void,
                                           &client_data);
                    debug!(~"before run tcp req loop");
                    run(test_loop);
                    debug!(~"after run tcp req loop");
                }
                else {
                   debug!(~"tcp_connect() failure");
                   assert!(false);
                }
            }
            else {
                debug!(~"tcp_init() failure");
                assert!(false);
            }
            loop_delete(test_loop);
        }
    }

    extern fn server_after_close_cb(handle: *libc::c_void) {
        debug!("SERVER server stream closed, should exit. h: %?",
                   handle);
    }

    extern fn client_stream_after_close_cb(handle: *libc::c_void) {
        unsafe {
            debug!(~"SERVER: closed client stream, now closing server stream");
            let client_data = get_data_for_uv_handle(
                handle) as
                *tcp_server_data;
            close((*client_data).server as *libc::c_void,
                          server_after_close_cb);
        }
    }

    extern fn after_server_resp_write(req: *uv_write_t) {
        unsafe {
            let client_stream_ptr =
                get_stream_handle_from_write_req(req);
            debug!(~"SERVER: resp sent... closing client stream");
            close(client_stream_ptr as *libc::c_void,
                          client_stream_after_close_cb)
        }
    }

    extern fn on_server_read_cb(client_stream_ptr: *uv_stream_t,
                               nread: libc::ssize_t,
                               buf: uv_buf_t) {
        unsafe {
            let nread = nread as int;
            if (nread > 0) {
                // we have data
                debug!("SERVER read: data! nread: %d", nread);

                // pull out the contents of the write from the client
                let buf_base = get_base_from_buf(buf);
                let buf_len = get_len_from_buf(buf) as uint;
                debug!("SERVER buf base: %u, len: %u, nread: %d",
                                buf_base as uint,
                                buf_len as uint,
                                nread);
                let bytes = vec::from_buf(buf_base, nread as uint);
                let request_str = str::from_bytes(bytes);

                let client_data = get_data_for_uv_handle(
                    client_stream_ptr as *libc::c_void) as *tcp_server_data;

                let server_kill_msg = copy (*client_data).server_kill_msg;
                let write_req = (*client_data).server_write_req;
                if str::contains(request_str, server_kill_msg) {
                    debug!(~"SERVER: client req contains kill_msg!");
                    debug!(~"SERVER: sending response to client");
                    read_stop(client_stream_ptr);
                    let server_chan = (*client_data).server_chan.clone();
                    server_chan.send(request_str);
                    let write_result = write(
                        write_req,
                        client_stream_ptr as *libc::c_void,
                        (*client_data).server_resp_buf,
                        after_server_resp_write);
                    debug!("SERVER: resp write result: %d",
                                write_result as int);
                    if (write_result != 0i32) {
                        debug!(~"bad result for server resp write()");
                        debug!(get_last_err_info(
                            get_loop_for_uv_handle(client_stream_ptr
                                as *libc::c_void)));
                        assert!(false);
                    }
                }
                else {
                    debug!(~"SERVER: client req !contain kill_msg!");
                }
            }
            else if (nread == -1) {
                // err .. possibly EOF
                debug!(~"read: eof!");
            }
            else {
                // nread == 0 .. do nothing, just free buf as below
                debug!(~"read: do nothing!");
            }
            // when we're done
            free_base_of_buf(buf);
            debug!(~"SERVER exiting on_read_cb");
        }
    }

    extern fn server_connection_cb(server_stream_ptr:
                                    *uv_stream_t,
                                  status: libc::c_int) {
        unsafe {
            debug!(~"client connecting!");
            let test_loop = get_loop_for_uv_handle(
                                   server_stream_ptr as *libc::c_void);
            if status != 0i32 {
                let err_msg = get_last_err_info(test_loop);
                debug!("server_connect_cb: non-zero status: %?",
                             err_msg);
                return;
            }
            let server_data = get_data_for_uv_handle(
                server_stream_ptr as *libc::c_void) as *tcp_server_data;
            let client_stream_ptr = (*server_data).client;
            let client_init_result = tcp_init(test_loop,
                                                      client_stream_ptr);
            set_data_for_uv_handle(
                client_stream_ptr as *libc::c_void,
                server_data as *libc::c_void);
            if (client_init_result == 0i32) {
                debug!(~"successfully initialized client stream");
                let accept_result = accept(server_stream_ptr as
                                                     *libc::c_void,
                                                   client_stream_ptr as
                                                     *libc::c_void);
                if (accept_result == 0i32) {
                    // start reading
                    let read_result = read_start(
                        client_stream_ptr as *uv_stream_t,
                                                         on_alloc_cb,
                                                         on_server_read_cb);
                    if (read_result == 0i32) {
                        debug!(~"successful server read start");
                    }
                    else {
                        debug!("server_connection_cb: bad read:%d",
                                        read_result as int);
                        assert!(false);
                    }
                }
                else {
                    debug!("server_connection_cb: bad accept: %d",
                                accept_result as int);
                    assert!(false);
                }
            }
            else {
                debug!("server_connection_cb: bad client init: %d",
                            client_init_result as int);
                assert!(false);
            }
        }
    }

    struct tcp_server_data {
        client: *uv_tcp_t,
        server: *uv_tcp_t,
        server_kill_msg: ~str,
        server_resp_buf: *~[uv_buf_t],
        server_chan: SharedChan<~str>,
        server_write_req: *uv_write_t,
    }

    struct async_handle_data {
        continue_chan: SharedChan<bool>,
    }

    extern fn async_close_cb(handle: *libc::c_void) {
        debug!("SERVER: closing async cb... h: %?",
                   handle);
    }

    extern fn continue_async_cb(async_handle: *uv_async_t,
                               status: libc::c_int) {
        unsafe {
            // once we're in the body of this callback,
            // the tcp server's loop is set up, so we
            // can continue on to let the tcp client
            // do its thang
            let data = get_data_for_uv_handle(
                async_handle as *libc::c_void) as *async_handle_data;
            let continue_chan = (*data).continue_chan.clone();
            let should_continue = status == 0i32;
            continue_chan.send(should_continue);
            close(async_handle as *libc::c_void, async_close_cb);
        }
    }

    fn impl_uv_tcp_server(server_ip: &str,
                          server_port: int,
                          kill_server_msg: ~str,
                          server_resp_msg: ~str,
                          server_chan: SharedChan<~str>,
                          continue_chan: SharedChan<bool>) {
        unsafe {
            let test_loop = loop_new();
            let tcp_server = tcp_t();
            let tcp_server_ptr: *uv_tcp_t = &tcp_server;

            let tcp_client = tcp_t();
            let tcp_client_ptr: *uv_tcp_t = &tcp_client;

            let server_write_req = write_t();
            let server_write_req_ptr: *uv_write_t = &server_write_req;

            let resp_str_bytes = str::to_bytes(server_resp_msg);
            let resp_msg_ptr: *u8 = vec::raw::to_ptr(resp_str_bytes);
            debug!("resp_msg ptr: %u", resp_msg_ptr as uint);
            let resp_msg = ~[
                buf_init(resp_msg_ptr, resp_str_bytes.len())
            ];

            let continue_async_handle = async_t();
            let continue_async_handle_ptr: *uv_async_t =
                &continue_async_handle;
            let async_data =
                async_handle_data { continue_chan: continue_chan };
            let async_data_ptr: *async_handle_data = &async_data;

            let server_data = tcp_server_data {
                client: tcp_client_ptr,
                server: tcp_server_ptr,
                server_kill_msg: kill_server_msg,
                server_resp_buf: &resp_msg,
                server_chan: server_chan,
                server_write_req: server_write_req_ptr
            };
            let server_data_ptr: *tcp_server_data = &server_data;
            set_data_for_uv_handle(tcp_server_ptr as *libc::c_void,
                                           server_data_ptr as *libc::c_void);

            // uv_tcp_init()
            let tcp_init_result = tcp_init(
                test_loop as *libc::c_void, tcp_server_ptr);
            if (tcp_init_result == 0i32) {
                let server_addr = ip4_addr(server_ip, server_port);
                // FIXME ref #2064
                let server_addr_ptr: *sockaddr_in = &server_addr;

                // uv_tcp_bind()
                let bind_result = tcp_bind(tcp_server_ptr, server_addr_ptr);
                if (bind_result == 0i32) {
                    debug!(~"successful uv_tcp_bind, listening");

                    // uv_listen()
                    let listen_result = listen(tcp_server_ptr as
                                                         *libc::c_void,
                                                       128i32,
                                                       server_connection_cb);
                    if (listen_result == 0i32) {
                        // let the test know it can set up the tcp server,
                        // now.. this may still present a race, not sure..
                        let async_result = async_init(test_loop,
                                           continue_async_handle_ptr,
                                           continue_async_cb);
                        if (async_result == 0i32) {
                            set_data_for_uv_handle(
                                continue_async_handle_ptr as *libc::c_void,
                                async_data_ptr as *libc::c_void);
                            async_send(continue_async_handle_ptr);
                            // uv_run()
                            run(test_loop);
                            debug!(~"server uv::run() has returned");
                        }
                        else {
                            debug!("uv_async_init failure: %d",
                                    async_result as int);
                            assert!(false);
                        }
                    }
                    else {
                        debug!("non-zero result on uv_listen: %d",
                                    listen_result as int);
                        assert!(false);
                    }
                }
                else {
                    debug!("non-zero result on uv_tcp_bind: %d",
                                bind_result as int);
                    assert!(false);
                }
            }
            else {
                debug!("non-zero result on uv_tcp_init: %d",
                            tcp_init_result as int);
                assert!(false);
            }
            loop_delete(test_loop);
        }
    }

    // this is the impl for a test that is (maybe) ran on a
    // per-platform/arch basis below
    pub fn impl_uv_tcp_server_and_request() {
        let bind_ip = ~"0.0.0.0";
        let request_ip = ~"127.0.0.1";
        let port = 8886;
        let kill_server_msg = ~"does a dog have buddha nature?";
        let server_resp_msg = ~"mu!";
        let (client_port, client_chan) = stream::<~str>();
        let client_chan = SharedChan::new(client_chan);
        let (server_port, server_chan) = stream::<~str>();
        let server_chan = SharedChan::new(server_chan);

        let (continue_port, continue_chan) = stream::<bool>();
        let continue_chan = SharedChan::new(continue_chan);

        let kill_server_msg_copy = copy kill_server_msg;
        let server_resp_msg_copy = copy server_resp_msg;
        do task::spawn_sched(task::ManualThreads(1)) {
            impl_uv_tcp_server(bind_ip, port,
                               copy kill_server_msg_copy,
                               copy server_resp_msg_copy,
                               server_chan.clone(),
                               continue_chan.clone());
        };

        // block until the server up is.. possibly a race?
        debug!(~"before receiving on server continue_port");
        continue_port.recv();
        debug!(~"received on continue port, set up tcp client");

        let kill_server_msg_copy = copy kill_server_msg;
        do task::spawn_sched(task::ManualThreads(1u)) {
            impl_uv_tcp_request(request_ip, port,
                               kill_server_msg_copy,
                               client_chan.clone());
        };

        let msg_from_client = server_port.recv();
        let msg_from_server = client_port.recv();

        assert!(str::contains(msg_from_client, kill_server_msg));
        assert!(str::contains(msg_from_server, server_resp_msg));
    }

    // FIXME don't run on fbsd or linux 32 bit(#2064)
    #[cfg(target_os="win32")]
    #[cfg(target_os="darwin")]
    #[cfg(target_os="linux")]
    #[cfg(target_os="android")]
    mod tcp_and_server_client_test {
        #[cfg(target_arch="x86_64")]
        mod impl64 {
            #[test]
            fn test_uv_ll_tcp_server_and_request() {
                unsafe {
                    super::super::impl_uv_tcp_server_and_request();
                }
            }
        }
        #[cfg(target_arch="x86")]
        #[cfg(target_arch="arm")]
        #[cfg(target_arch="mips")]
        mod impl32 {
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            fn test_uv_ll_tcp_server_and_request() {
                unsafe {
                    super::super::impl_uv_tcp_server_and_request();
                }
            }
        }
    }

    fn struct_size_check_common<TStruct>(t_name: ~str,
                                         foreign_size: libc::c_uint) {
        let rust_size = sys::size_of::<TStruct>();
        let sizes_match = foreign_size as uint == rust_size;
        if !sizes_match {
            let output = fmt!(
                "STRUCT_SIZE FAILURE: %s -- actual: %u expected: %u",
                t_name, rust_size, foreign_size as uint);
            debug!(output);
        }
        assert!(sizes_match);
    }

    // struct size tests
    #[test]
    fn test_uv_ll_struct_size_uv_tcp_t() {
        unsafe {
            struct_size_check_common::<uv_tcp_t>(
                ~"uv_tcp_t",
                super::rust_uv_helper_uv_tcp_t_size()
            );
        }
    }
    #[test]
    fn test_uv_ll_struct_size_uv_connect_t() {
        unsafe {
            struct_size_check_common::<uv_connect_t>(
                ~"uv_connect_t",
                super::rust_uv_helper_uv_connect_t_size()
            );
        }
    }
    #[test]
    fn test_uv_ll_struct_size_uv_buf_t() {
        unsafe {
            struct_size_check_common::<uv_buf_t>(
                ~"uv_buf_t",
                super::rust_uv_helper_uv_buf_t_size()
            );
        }
    }
    #[test]
    fn test_uv_ll_struct_size_uv_write_t() {
        unsafe {
            struct_size_check_common::<uv_write_t>(
                ~"uv_write_t",
                super::rust_uv_helper_uv_write_t_size()
            );
        }
    }

    #[test]
    fn test_uv_ll_struct_size_sockaddr_in() {
        unsafe {
            struct_size_check_common::<sockaddr_in>(
                ~"sockaddr_in",
                super::rust_uv_helper_sockaddr_in_size()
            );
        }
    }
    #[test]
    fn test_uv_ll_struct_size_sockaddr_in6() {
        unsafe {
            let foreign_handle_size =
                super::rust_uv_helper_sockaddr_in6_size();
            let rust_handle_size = sys::size_of::<sockaddr_in6>();
            let output = fmt!("sockaddr_in6 -- foreign: %u rust: %u",
                              foreign_handle_size as uint, rust_handle_size);
            debug!(output);
            // FIXME #1645 .. rust appears to pad structs to the nearest
            // byte..?
            // .. can't get the uv::ll::sockaddr_in6 to == 28 :/
            // .. so the type always appears to be 32 in size.. which is
            // good, i guess.. better too big than too little
            assert!((4u+foreign_handle_size as uint) ==
                rust_handle_size);
        }
    }
    #[test]
    #[ignore(reason = "questionable size calculations")]
    fn test_uv_ll_struct_size_addr_in() {
        unsafe {
            let foreign_handle_size =
                super::rust_uv_helper_addr_in_size();
            let rust_handle_size = sys::size_of::<addr_in>();
            let output = fmt!("addr_in -- foreign: %u rust: %u",
                              foreign_handle_size as uint, rust_handle_size);
            debug!(output);
            // FIXME #1645 .. see note above about struct padding
            assert!((4u+foreign_handle_size as uint) ==
                rust_handle_size);
        }
    }

    #[test]
    fn test_uv_ll_struct_size_uv_async_t() {
        unsafe {
            struct_size_check_common::<uv_async_t>(
                ~"uv_async_t",
                super::rust_uv_helper_uv_async_t_size()
            );
        }
    }

    #[test]
    fn test_uv_ll_struct_size_uv_timer_t() {
        unsafe {
            struct_size_check_common::<uv_timer_t>(
                ~"uv_timer_t",
                super::rust_uv_helper_uv_timer_t_size()
            );
        }
    }

    #[test]
    #[ignore(cfg(target_os = "win32"))]
    fn test_uv_ll_struct_size_uv_getaddrinfo_t() {
        unsafe {
            struct_size_check_common::<uv_getaddrinfo_t>(
                ~"uv_getaddrinfo_t",
                super::rust_uv_helper_uv_getaddrinfo_t_size()
            );
        }
    }
    #[test]
    #[ignore(cfg(target_os = "macos"))]
    #[ignore(cfg(target_os = "win32"))]
    fn test_uv_ll_struct_size_addrinfo() {
        unsafe {
            struct_size_check_common::<uv_timer_t>(
                ~"addrinfo",
                super::rust_uv_helper_uv_timer_t_size()
            );
        }
    }
}
