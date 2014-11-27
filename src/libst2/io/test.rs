// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Various utility functions useful for writing I/O tests */

#![macro_escape]

use libc;
use os;
use prelude::*;
use std::io::net::ip::*;
use sync::atomic::{AtomicUint, INIT_ATOMIC_UINT, Relaxed};

/// Get a port number, starting at 9600, for use in tests
pub fn next_test_port() -> u16 { unimplemented!() }

/// Get a temporary path which could be the location of a unix socket
pub fn next_test_unix() -> Path { unimplemented!() }

/// Get a unique IPv4 localhost:port pair starting at 9600
pub fn next_test_ip4() -> SocketAddr { unimplemented!() }

/// Get a unique IPv6 localhost:port pair starting at 9600
pub fn next_test_ip6() -> SocketAddr { unimplemented!() }

/*
XXX: Welcome to MegaHack City.

The bots run multiple builds at the same time, and these builds
all want to use ports. This function figures out which workspace
it is running in and assigns a port range based on it.
*/
fn base_port() -> u16 { unimplemented!() }

/// Raises the file descriptor limit when running tests if necessary
pub fn raise_fd_limit() { unimplemented!() }

#[cfg(target_os="macos")]
#[allow(non_camel_case_types)]
mod darwin_fd_limit {
    /*!
     * darwin_fd_limit exists to work around an issue where launchctl on Mac OS X defaults the
     * rlimit maxfiles to 256/unlimited. The default soft limit of 256 ends up being far too low
     * for our multithreaded scheduler testing, depending on the number of cores available.
     *
     * This fixes issue #7772.
     */

    use libc;
    type rlim_t = libc::uint64_t;
    #[repr(C)]
    struct rlimit {
        rlim_cur: rlim_t,
        rlim_max: rlim_t
    }
    extern {
        // name probably doesn't need to be mut, but the C function doesn't specify const
        fn sysctl(name: *mut libc::c_int, namelen: libc::c_uint,
                  oldp: *mut libc::c_void, oldlenp: *mut libc::size_t,
                  newp: *mut libc::c_void, newlen: libc::size_t) -> libc::c_int;
        fn getrlimit(resource: libc::c_int, rlp: *mut rlimit) -> libc::c_int;
        fn setrlimit(resource: libc::c_int, rlp: *const rlimit) -> libc::c_int;
    }
    static CTL_KERN: libc::c_int = 1;
    static KERN_MAXFILESPERPROC: libc::c_int = 29;
    static RLIMIT_NOFILE: libc::c_int = 8;

    pub unsafe fn raise_fd_limit() { unimplemented!() }
}

#[cfg(not(target_os="macos"))]
mod darwin_fd_limit {
    pub unsafe fn raise_fd_limit() { unimplemented!() }
}
