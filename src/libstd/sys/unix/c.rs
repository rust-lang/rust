// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! C definitions used by libnative that don't belong in liblibc

#![allow(dead_code)]
#![allow(non_camel_case_types)]

pub use self::select::fd_set;
pub use self::signal::{sigaction, siginfo, sigset_t};
pub use self::signal::{SA_ONSTACK, SA_RESTART, SA_RESETHAND, SA_NOCLDSTOP};
pub use self::signal::{SA_NODEFER, SA_NOCLDWAIT, SA_SIGINFO, SIGCHLD};

use libc;

#[cfg(any(target_os = "macos",
          target_os = "ios",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "openbsd"))]
pub const FIONBIO: libc::c_ulong = 0x8004667e;
#[cfg(any(all(target_os = "linux",
              any(target_arch = "x86",
                  target_arch = "x86_64",
                  target_arch = "arm",
                  target_arch = "aarch64")),
          target_os = "android"))]
pub const FIONBIO: libc::c_ulong = 0x5421;
#[cfg(all(target_os = "linux",
          any(target_arch = "mips",
              target_arch = "mipsel",
              target_arch = "powerpc")))]
pub const FIONBIO: libc::c_ulong = 0x667e;

#[cfg(any(target_os = "macos",
          target_os = "ios",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "openbsd"))]
pub const FIOCLEX: libc::c_ulong = 0x20006601;
#[cfg(any(all(target_os = "linux",
              any(target_arch = "x86",
                  target_arch = "x86_64",
                  target_arch = "arm",
                  target_arch = "aarch64")),
          target_os = "android"))]
pub const FIOCLEX: libc::c_ulong = 0x5451;
#[cfg(all(target_os = "linux",
          any(target_arch = "mips",
              target_arch = "mipsel",
              target_arch = "powerpc")))]
pub const FIOCLEX: libc::c_ulong = 0x6601;

#[cfg(any(target_os = "macos",
          target_os = "ios",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "openbsd"))]
pub const MSG_DONTWAIT: libc::c_int = 0x80;
#[cfg(any(target_os = "linux", target_os = "android"))]
pub const MSG_DONTWAIT: libc::c_int = 0x40;

pub const WNOHANG: libc::c_int = 1;

#[cfg(target_os = "linux")]
pub const _SC_GETPW_R_SIZE_MAX: libc::c_int = 70;
#[cfg(any(target_os = "macos",
          target_os = "freebsd",
          target_os = "dragonfly"))]
pub const _SC_GETPW_R_SIZE_MAX: libc::c_int = 71;
#[cfg(any(target_os = "bitrig",
          target_os = "openbsd"))]
pub const _SC_GETPW_R_SIZE_MAX: libc::c_int = 101;
#[cfg(target_os = "android")]
pub const _SC_GETPW_R_SIZE_MAX: libc::c_int = 0x0048;

#[repr(C)]
#[cfg(target_os = "linux")]
pub struct passwd {
    pub pw_name: *mut libc::c_char,
    pub pw_passwd: *mut libc::c_char,
    pub pw_uid: libc::uid_t,
    pub pw_gid: libc::gid_t,
    pub pw_gecos: *mut libc::c_char,
    pub pw_dir: *mut libc::c_char,
    pub pw_shell: *mut libc::c_char,
}

#[repr(C)]
#[cfg(any(target_os = "macos",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "openbsd"))]
pub struct passwd {
    pub pw_name: *mut libc::c_char,
    pub pw_passwd: *mut libc::c_char,
    pub pw_uid: libc::uid_t,
    pub pw_gid: libc::gid_t,
    pub pw_change: libc::time_t,
    pub pw_class: *mut libc::c_char,
    pub pw_gecos: *mut libc::c_char,
    pub pw_dir: *mut libc::c_char,
    pub pw_shell: *mut libc::c_char,
    pub pw_expire: libc::time_t,
}

#[repr(C)]
#[cfg(target_os = "android")]
pub struct passwd {
    pub pw_name: *mut libc::c_char,
    pub pw_passwd: *mut libc::c_char,
    pub pw_uid: libc::uid_t,
    pub pw_gid: libc::gid_t,
    pub pw_dir: *mut libc::c_char,
    pub pw_shell: *mut libc::c_char,
}

extern {
    pub fn gettimeofday(timeval: *mut libc::timeval,
                        tzp: *mut libc::c_void) -> libc::c_int;
    pub fn select(nfds: libc::c_int,
                  readfds: *mut fd_set,
                  writefds: *mut fd_set,
                  errorfds: *mut fd_set,
                  timeout: *mut libc::timeval) -> libc::c_int;
    pub fn getsockopt(sockfd: libc::c_int,
                      level: libc::c_int,
                      optname: libc::c_int,
                      optval: *mut libc::c_void,
                      optlen: *mut libc::socklen_t) -> libc::c_int;
    pub fn ioctl(fd: libc::c_int, req: libc::c_ulong, ...) -> libc::c_int;


    pub fn waitpid(pid: libc::pid_t, status: *mut libc::c_int,
                   options: libc::c_int) -> libc::pid_t;

    pub fn sigaction(signum: libc::c_int,
                     act: *const sigaction,
                     oldact: *mut sigaction) -> libc::c_int;

    pub fn sigaddset(set: *mut sigset_t, signum: libc::c_int) -> libc::c_int;
    pub fn sigdelset(set: *mut sigset_t, signum: libc::c_int) -> libc::c_int;
    pub fn sigemptyset(set: *mut sigset_t) -> libc::c_int;

    #[cfg(not(target_os = "ios"))]
    pub fn getpwuid_r(uid: libc::uid_t,
                      pwd: *mut passwd,
                      buf: *mut libc::c_char,
                      buflen: libc::size_t,
                      result: *mut *mut passwd) -> libc::c_int;

    pub fn utimes(filename: *const libc::c_char,
                  times: *const libc::timeval) -> libc::c_int;
    pub fn gai_strerror(errcode: libc::c_int) -> *const libc::c_char;
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod select {
    pub const FD_SETSIZE: usize = 1024;

    #[repr(C)]
    pub struct fd_set {
        fds_bits: [i32; (FD_SETSIZE / 32)]
    }

    pub fn fd_set(set: &mut fd_set, fd: i32) {
        set.fds_bits[(fd / 32) as usize] |= 1 << ((fd % 32) as usize);
    }
}

#[cfg(any(target_os = "android",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "openbsd",
          target_os = "linux"))]
mod select {
    use usize;
    use libc;

    pub const FD_SETSIZE: usize = 1024;

    #[repr(C)]
    pub struct fd_set {
        // FIXME: shouldn't this be a c_ulong?
        fds_bits: [libc::uintptr_t; (FD_SETSIZE / usize::BITS)]
    }

    pub fn fd_set(set: &mut fd_set, fd: i32) {
        let fd = fd as usize;
        set.fds_bits[fd / usize::BITS] |= 1 << (fd % usize::BITS);
    }
}

#[cfg(any(all(target_os = "linux",
              any(target_arch = "x86",
                  target_arch = "x86_64",
                  target_arch = "arm",
                  target_arch = "aarch64")),
          target_os = "android"))]
mod signal {
    use libc;

    pub const SA_NOCLDSTOP: libc::c_ulong = 0x00000001;
    pub const SA_NOCLDWAIT: libc::c_ulong = 0x00000002;
    pub const SA_NODEFER: libc::c_ulong = 0x40000000;
    pub const SA_ONSTACK: libc::c_ulong = 0x08000000;
    pub const SA_RESETHAND: libc::c_ulong = 0x80000000;
    pub const SA_RESTART: libc::c_ulong = 0x10000000;
    pub const SA_SIGINFO: libc::c_ulong = 0x00000004;
    pub const SIGCHLD: libc::c_int = 17;

    // This definition is not as accurate as it could be, {pid, uid, status} is
    // actually a giant union. Currently we're only interested in these fields,
    // however.
    #[repr(C)]
    pub struct siginfo {
        si_signo: libc::c_int,
        si_errno: libc::c_int,
        si_code: libc::c_int,
        pub pid: libc::pid_t,
        pub uid: libc::uid_t,
        pub status: libc::c_int,
    }

    #[repr(C)]
    pub struct sigaction {
        pub sa_handler: extern fn(libc::c_int),
        pub sa_mask: sigset_t,
        pub sa_flags: libc::c_ulong,
        sa_restorer: *mut libc::c_void,
    }

    unsafe impl ::marker::Send for sigaction { }
    unsafe impl ::marker::Sync for sigaction { }

    #[repr(C)]
    #[cfg(target_pointer_width = "32")]
    pub struct sigset_t {
        __val: [libc::c_ulong; 32],
    }

    #[repr(C)]
    #[cfg(target_pointer_width = "64")]
    pub struct sigset_t {
        __val: [libc::c_ulong; 16],
    }
}

#[cfg(all(target_os = "linux",
          any(target_arch = "mips",
              target_arch = "mipsel",
              target_arch = "powerpc")))]
mod signal {
    use libc;

    pub const SA_NOCLDSTOP: libc::c_ulong = 0x00000001;
    pub const SA_NOCLDWAIT: libc::c_ulong = 0x00010000;
    pub const SA_NODEFER: libc::c_ulong = 0x40000000;
    pub const SA_ONSTACK: libc::c_ulong = 0x08000000;
    pub const SA_RESETHAND: libc::c_ulong = 0x80000000;
    pub const SA_RESTART: libc::c_ulong = 0x10000000;
    pub const SA_SIGINFO: libc::c_ulong = 0x00000008;
    pub const SIGCHLD: libc::c_int = 18;

    // This definition is not as accurate as it could be, {pid, uid, status} is
    // actually a giant union. Currently we're only interested in these fields,
    // however.
    #[repr(C)]
    pub struct siginfo {
        si_signo: libc::c_int,
        si_code: libc::c_int,
        si_errno: libc::c_int,
        pub pid: libc::pid_t,
        pub uid: libc::uid_t,
        pub status: libc::c_int,
    }

    #[repr(C)]
    pub struct sigaction {
        pub sa_flags: libc::c_uint,
        pub sa_handler: extern fn(libc::c_int),
        pub sa_mask: sigset_t,
        sa_restorer: *mut libc::c_void,
        sa_resv: [libc::c_int; 1],
    }

    unsafe impl ::marker::Send for sigaction { }
    unsafe impl ::marker::Sync for sigaction { }

    #[repr(C)]
    pub struct sigset_t {
        __val: [libc::c_ulong; 32],
    }
}

#[cfg(any(target_os = "macos",
          target_os = "ios",
          target_os = "freebsd",
          target_os = "dragonfly"))]
mod signal {
    use libc;

    pub const SA_ONSTACK: libc::c_int = 0x0001;
    pub const SA_RESTART: libc::c_int = 0x0002;
    pub const SA_RESETHAND: libc::c_int = 0x0004;
    pub const SA_NOCLDSTOP: libc::c_int = 0x0008;
    pub const SA_NODEFER: libc::c_int = 0x0010;
    pub const SA_NOCLDWAIT: libc::c_int = 0x0020;
    pub const SA_SIGINFO: libc::c_int = 0x0040;
    pub const SIGCHLD: libc::c_int = 20;

    #[cfg(any(target_os = "macos",
              target_os = "ios"))]
    pub type sigset_t = u32;
    #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
    #[repr(C)]
    pub struct sigset_t {
        bits: [u32; 4],
    }

    // This structure has more fields, but we're not all that interested in
    // them.
    #[repr(C)]
    pub struct siginfo {
        pub si_signo: libc::c_int,
        pub si_errno: libc::c_int,
        pub si_code: libc::c_int,
        pub pid: libc::pid_t,
        pub uid: libc::uid_t,
        pub status: libc::c_int,
    }

    #[repr(C)]
    pub struct sigaction {
        pub sa_handler: extern fn(libc::c_int),
        pub sa_flags: libc::c_int,
        pub sa_mask: sigset_t,
    }
}

#[cfg(any(target_os = "bitrig", target_os = "openbsd"))]
mod signal {
    use libc;

    pub const SA_ONSTACK: libc::c_int = 0x0001;
    pub const SA_RESTART: libc::c_int = 0x0002;
    pub const SA_RESETHAND: libc::c_int = 0x0004;
    pub const SA_NOCLDSTOP: libc::c_int = 0x0008;
    pub const SA_NODEFER: libc::c_int = 0x0010;
    pub const SA_NOCLDWAIT: libc::c_int = 0x0020;
    pub const SA_SIGINFO: libc::c_int = 0x0040;
    pub const SIGCHLD: libc::c_int = 20;

    pub type sigset_t = libc::c_uint;

    // This structure has more fields, but we're not all that interested in
    // them.
    #[repr(C)]
    pub struct siginfo {
        pub si_signo: libc::c_int,
        pub si_code: libc::c_int,
        pub si_errno: libc::c_int,
        // FIXME: Bitrig has a crazy union here in the siginfo, I think this
        // layout will still work tho.  The status might be off by the size of
        // a clock_t by my reading, but we can fix this later.
        pub pid: libc::pid_t,
        pub uid: libc::uid_t,
        pub status: libc::c_int,
    }

    #[repr(C)]
    pub struct sigaction {
        pub sa_handler: extern fn(libc::c_int),
        pub sa_mask: sigset_t,
        pub sa_flags: libc::c_int,
    }
}
