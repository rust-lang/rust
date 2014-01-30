// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_escape];

use os;
use prelude::*;
use rand;
use rand::Rng;
use std::io::net::ip::*;
use sync::atomics::{AtomicUint, INIT_ATOMIC_UINT, Relaxed};

macro_rules! iotest (
    { fn $name:ident() $b:block $($a:attr)* } => (
        mod $name {
            #[allow(unused_imports)];

            use super::super::*;
            use super::*;
            use io;
            use prelude::*;
            use io::*;
            use io::fs::*;
            use io::test::*;
            use io::net::tcp::*;
            use io::net::ip::*;
            use io::net::udp::*;
            #[cfg(unix)]
            use io::net::unix::*;
            use io::timer::*;
            use io::process::*;
            use str;
            use util;

            fn f() $b

            $($a)* #[test] fn green() { f() }
            $($a)* #[test] fn native() {
                use native;
                let (p, c) = Chan::new();
                native::task::spawn(proc() { c.send(f()) });
                p.recv();
            }
        }
    )
)

/// Get a port number, starting at 9600, for use in tests
pub fn next_test_port() -> u16 {
    static mut next_offset: AtomicUint = INIT_ATOMIC_UINT;
    unsafe {
        base_port() + next_offset.fetch_add(1, Relaxed) as u16
    }
}

/// Get a temporary path which could be the location of a unix socket
pub fn next_test_unix() -> Path {
    if cfg!(unix) {
        os::tmpdir().join(rand::task_rng().gen_ascii_str(20))
    } else {
        Path::new(r"\\.\pipe\" + rand::task_rng().gen_ascii_str(20))
    }
}

/// Get a unique IPv4 localhost:port pair starting at 9600
pub fn next_test_ip4() -> SocketAddr {
    SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: next_test_port() }
}

/// Get a unique IPv6 localhost:port pair starting at 9600
pub fn next_test_ip6() -> SocketAddr {
    SocketAddr { ip: Ipv6Addr(0, 0, 0, 0, 0, 0, 0, 1), port: next_test_port() }
}

/*
XXX: Welcome to MegaHack City.

The bots run multiple builds at the same time, and these builds
all want to use ports. This function figures out which workspace
it is running in and assigns a port range based on it.
*/
fn base_port() -> u16 {

    let base = 9600u16;
    let range = 1000u16;

    let bases = [
        ("32-opt", base + range * 1),
        ("32-nopt", base + range * 2),
        ("64-opt", base + range * 3),
        ("64-nopt", base + range * 4),
        ("64-opt-vg", base + range * 5),
        ("all-opt", base + range * 6),
        ("snap3", base + range * 7),
        ("dist", base + range * 8)
    ];

    // FIXME (#9639): This needs to handle non-utf8 paths
    let path = os::getcwd();
    let path_s = path.as_str().unwrap();

    let mut final_base = base;

    for &(dir, base) in bases.iter() {
        if path_s.contains(dir) {
            final_base = base;
            break;
        }
    }

    return final_base;
}

pub fn raise_fd_limit() {
    unsafe { darwin_fd_limit::raise_fd_limit() }
}

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
    struct rlimit {
        rlim_cur: rlim_t,
        rlim_max: rlim_t
    }
    #[nolink]
    extern {
        // name probably doesn't need to be mut, but the C function doesn't specify const
        fn sysctl(name: *mut libc::c_int, namelen: libc::c_uint,
                  oldp: *mut libc::c_void, oldlenp: *mut libc::size_t,
                  newp: *mut libc::c_void, newlen: libc::size_t) -> libc::c_int;
        fn getrlimit(resource: libc::c_int, rlp: *mut rlimit) -> libc::c_int;
        fn setrlimit(resource: libc::c_int, rlp: *rlimit) -> libc::c_int;
    }
    static CTL_KERN: libc::c_int = 1;
    static KERN_MAXFILESPERPROC: libc::c_int = 29;
    static RLIMIT_NOFILE: libc::c_int = 8;

    pub unsafe fn raise_fd_limit() {
        // The strategy here is to fetch the current resource limits, read the kern.maxfilesperproc
        // sysctl value, and bump the soft resource limit for maxfiles up to the sysctl value.
        use ptr::{to_unsafe_ptr, to_mut_unsafe_ptr, mut_null};
        use mem::size_of_val;
        use os::last_os_error;

        // Fetch the kern.maxfilesperproc value
        let mut mib: [libc::c_int, ..2] = [CTL_KERN, KERN_MAXFILESPERPROC];
        let mut maxfiles: libc::c_int = 0;
        let mut size: libc::size_t = size_of_val(&maxfiles) as libc::size_t;
        if sysctl(to_mut_unsafe_ptr(&mut mib[0]), 2,
                  to_mut_unsafe_ptr(&mut maxfiles) as *mut libc::c_void,
                  to_mut_unsafe_ptr(&mut size),
                  mut_null(), 0) != 0 {
            let err = last_os_error();
            error!("raise_fd_limit: error calling sysctl: {}", err);
            return;
        }

        // Fetch the current resource limits
        let mut rlim = rlimit{rlim_cur: 0, rlim_max: 0};
        if getrlimit(RLIMIT_NOFILE, to_mut_unsafe_ptr(&mut rlim)) != 0 {
            let err = last_os_error();
            error!("raise_fd_limit: error calling getrlimit: {}", err);
            return;
        }

        // Bump the soft limit to the smaller of kern.maxfilesperproc and the hard limit
        rlim.rlim_cur = ::cmp::min(maxfiles as rlim_t, rlim.rlim_max);

        // Set our newly-increased resource limit
        if setrlimit(RLIMIT_NOFILE, to_unsafe_ptr(&rlim)) != 0 {
            let err = last_os_error();
            error!("raise_fd_limit: error calling setrlimit: {}", err);
            return;
        }
    }
}

#[cfg(not(target_os="macos"))]
mod darwin_fd_limit {
    pub unsafe fn raise_fd_limit() {}
}
