// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! C definitions used by std::sys that don't belong in liblibc

// These are definitions sufficient for the users in this directory.
// This is not a general-purpose binding to this functionality, and in
// some cases (notably the definition of siginfo_t), we intentionally
// have incomplete bindings so that we don't need to fight with unions.
//
// Note that these types need to match the definitions from the platform
// libc (currently glibc on Linux), not the kernel definitions / the
// syscall interface.  This has a few weirdnesses, like glibc's sigset_t
// being 1024 bits on all platforms. If you're adding a new GNU/Linux
// port, check glibc's sysdeps/unix/sysv/linux, not the kernel headers.

#![allow(dead_code)]
#![allow(non_camel_case_types)]

pub use self::signal_os::*;
use libc;

#[cfg(any(target_os = "macos",
          target_os = "ios",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "netbsd",
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

#[cfg(target_env = "newlib")]
pub const FD_CLOEXEC: libc::c_int = 1;
#[cfg(target_env = "newlib")]
pub const F_GETFD: libc::c_int = 1;
#[cfg(target_env = "newlib")]
pub const F_SETFD: libc::c_int = 2;

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
#[cfg(target_os = "netbsd")]
pub const _SC_GETPW_R_SIZE_MAX: libc::c_int = 48;
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
#[cfg(target_env = "newlib")]
pub struct passwd {
    pub pw_name: *mut libc::c_char,
    pub pw_passwd: *mut libc::c_char,
    pub pw_uid: libc::uid_t,
    pub pw_gid: libc::gid_t,
    pub pw_comment: *mut libc::c_char,
    pub pw_gecos: *mut libc::c_char,
    pub pw_dir: *mut libc::c_char,
    pub pw_shell: *mut libc::c_char,
}

#[repr(C)]
#[cfg(any(target_os = "macos",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "netbsd",
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

pub fn page_size() -> usize {
    unsafe {
        libc::sysconf(libc::_SC_PAGESIZE) as usize
    }
}

// This is really a function pointer (or a union of multiple function
// pointers), except for constants like SIG_DFL.
pub type sighandler_t = *mut libc::c_void;

pub const SIG_DFL: sighandler_t = 0 as sighandler_t;
pub const SIG_ERR: sighandler_t = !0 as sighandler_t;

extern {
    pub fn getsockopt(sockfd: libc::c_int,
                      level: libc::c_int,
                      optname: libc::c_int,
                      optval: *mut libc::c_void,
                      optlen: *mut libc::socklen_t) -> libc::c_int;
    #[cfg(not(target_env = "newlib"))]
    pub fn ioctl(fd: libc::c_int, req: libc::c_ulong, ...) -> libc::c_int;
    #[cfg(target_env = "newlib")]
    pub fn fnctl(fd: libc::c_int, req: libc::c_int, ...) -> libc::c_int;


    pub fn waitpid(pid: libc::pid_t, status: *mut libc::c_int,
                   options: libc::c_int) -> libc::pid_t;

    pub fn raise(signum: libc::c_int) -> libc::c_int;

    #[cfg_attr(target_os = "netbsd", link_name = "__sigaction14")]
    pub fn sigaction(signum: libc::c_int,
                     act: *const sigaction,
                     oldact: *mut sigaction) -> libc::c_int;

    #[cfg_attr(target_os = "netbsd", link_name = "__sigaltstack14")]
    #[cfg(not(target_env = "newlib"))]
    pub fn sigaltstack(ss: *const sigaltstack,
                       oss: *mut sigaltstack) -> libc::c_int;

    #[cfg(not(target_os = "android"))]
    #[cfg_attr(target_os = "netbsd", link_name = "__sigemptyset14")]
    pub fn sigemptyset(set: *mut sigset_t) -> libc::c_int;

    pub fn pthread_sigmask(how: libc::c_int, set: *const sigset_t,
                           oldset: *mut sigset_t) -> libc::c_int;

    #[cfg(not(target_os = "ios"))]
    #[cfg_attr(target_os = "netbsd", link_name = "__getpwuid_r50")]
    pub fn getpwuid_r(uid: libc::uid_t,
                      pwd: *mut passwd,
                      buf: *mut libc::c_char,
                      buflen: libc::size_t,
                      result: *mut *mut passwd) -> libc::c_int;

    #[cfg_attr(target_os = "netbsd", link_name = "__utimes50")]
    pub fn utimes(filename: *const libc::c_char,
                  times: *const libc::timeval) -> libc::c_int;
    pub fn gai_strerror(errcode: libc::c_int) -> *const libc::c_char;
    /// Newlib has this, but only for Cygwin.
    #[cfg(not(target_os = "nacl"))]
    pub fn setgroups(ngroups: libc::c_int,
                     ptr: *const libc::c_void) -> libc::c_int;
    pub fn realpath(pathname: *const libc::c_char, resolved: *mut libc::c_char)
                    -> *mut libc::c_char;
}

// Ugh. This is only available as an inline until Android API 21.
#[cfg(target_os = "android")]
pub unsafe fn sigemptyset(set: *mut sigset_t) -> libc::c_int {
    use intrinsics;
    intrinsics::write_bytes(set, 0, 1);
    return 0;
}

#[cfg(any(target_os = "linux",
          target_os = "android"))]
mod signal_os {
    pub use self::arch::{SA_ONSTACK, SA_SIGINFO, SIGBUS, SIG_SETMASK,
                         sigaction, sigaltstack};
    use libc;

    #[cfg(any(target_arch = "x86",
              target_arch = "x86_64",
              target_arch = "arm",
              target_arch = "mips",
              target_arch = "mipsel"))]
    pub const SIGSTKSZ: libc::size_t = 8192;

    // This is smaller on musl and Android, but no harm in being generous.
    #[cfg(any(target_arch = "aarch64",
              target_arch = "powerpc"))]
    pub const SIGSTKSZ: libc::size_t = 16384;

    // This definition is intentionally a subset of the C structure: the
    // fields after si_code are actually a giant union. We're only
    // interested in si_addr for this module, though.
    #[repr(C)]
    pub struct siginfo {
        _signo: libc::c_int,
        _errno: libc::c_int,
        _code: libc::c_int,
        // This structure will need extra padding here for MIPS64.
        pub si_addr: *mut libc::c_void
    }

    #[cfg(all(target_os = "linux", target_pointer_width = "32"))]
    #[repr(C)]
    pub struct sigset_t {
        __val: [libc::c_ulong; 32],
    }

    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[repr(C)]
    pub struct sigset_t {
        __val: [libc::c_ulong; 16],
    }

    // Android for MIPS has a 128-bit sigset_t, but we don't currently
    // support it. Android for AArch64 technically has a structure of a
    // single ulong.
    #[cfg(target_os = "android")]
    pub type sigset_t = libc::c_ulong;

    #[cfg(any(target_arch = "x86",
              target_arch = "x86_64",
              target_arch = "powerpc",
              target_arch = "arm",
              target_arch = "aarch64"))]
    mod arch {
        use libc;
        use super::super::sighandler_t;
        use super::sigset_t;

        pub const SA_ONSTACK: libc::c_ulong = 0x08000000;
        pub const SA_SIGINFO: libc::c_ulong = 0x00000004;

        pub const SIGBUS: libc::c_int = 7;

        pub const SIG_SETMASK: libc::c_int = 2;

        #[cfg(target_os = "linux")]
        #[repr(C)]
        pub struct sigaction {
            pub sa_sigaction: sighandler_t,
            pub sa_mask: sigset_t,
            pub sa_flags: libc::c_ulong,
            _restorer: *mut libc::c_void,
        }

        #[cfg(all(target_os = "android", target_pointer_width = "32"))]
        #[repr(C)]
        pub struct sigaction {
            pub sa_sigaction: sighandler_t,
            pub sa_flags: libc::c_ulong,
            _restorer: *mut libc::c_void,
            pub sa_mask: sigset_t,
        }

        #[cfg(all(target_os = "android", target_pointer_width = "64"))]
        #[repr(C)]
        pub struct sigaction {
            pub sa_flags: libc::c_uint,
            pub sa_sigaction: sighandler_t,
            pub sa_mask: sigset_t,
            _restorer: *mut libc::c_void,
        }

        #[repr(C)]
        pub struct sigaltstack {
            pub ss_sp: *mut libc::c_void,
            pub ss_flags: libc::c_int,
            pub ss_size: libc::size_t
        }
    }

    #[cfg(any(target_arch = "mips",
              target_arch = "mipsel"))]
    mod arch {
        use libc;
        use super::super::sighandler_t;
        use super::sigset_t;

        pub const SA_ONSTACK: libc::c_ulong = 0x08000000;
        pub const SA_SIGINFO: libc::c_ulong = 0x00000008;

        pub const SIGBUS: libc::c_int = 10;

        pub const SIG_SETMASK: libc::c_int = 3;

        #[cfg(all(target_os = "linux", not(target_env = "musl")))]
        #[repr(C)]
        pub struct sigaction {
            pub sa_flags: libc::c_uint,
            pub sa_sigaction: sighandler_t,
            pub sa_mask: sigset_t,
            _restorer: *mut libc::c_void,
            _resv: [libc::c_int; 1],
        }

        #[cfg(target_env = "musl")]
        #[repr(C)]
        pub struct sigaction {
            pub sa_sigaction: sighandler_t,
            pub sa_mask: sigset_t,
            pub sa_flags: libc::c_ulong,
            _restorer: *mut libc::c_void,
        }

        #[cfg(target_os = "android")]
        #[repr(C)]
        pub struct sigaction {
            pub sa_flags: libc::c_uint,
            pub sa_sigaction: sighandler_t,
            pub sa_mask: sigset_t,
        }

        #[repr(C)]
        pub struct sigaltstack {
            pub ss_sp: *mut libc::c_void,
            pub ss_size: libc::size_t,
            pub ss_flags: libc::c_int,
        }
    }
}

/// Note: Although the signal functions are defined on NaCl, they always fail.
/// Also, this could be cfg-ed on newlib instead of nacl, but these structures
/// can differ depending on the platform, so I've played it safe here.
#[cfg(target_os = "nacl")]
mod signal_os {
    use libc;

    pub static SA_NOCLDSTOP: libc::c_ulong = 1;
    pub static SA_SIGINFO:   libc::c_ulong = 2;

    pub type sigset_t = libc::c_ulong;
    #[repr(C)]
    pub struct sigaction {
        pub sa_flags: libc::c_int,
        pub sa_mask:  sigset_t,
        pub handler:  extern fn(libc::c_int),
    }
}

#[cfg(any(target_os = "macos",
          target_os = "ios",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "netbsd",
          target_os = "openbsd"))]
mod signal_os {
    use libc;
    use super::sighandler_t;

    pub const SA_ONSTACK: libc::c_int = 0x0001;
    pub const SA_SIGINFO: libc::c_int = 0x0040;

    pub const SIGBUS: libc::c_int = 10;

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub const SIGSTKSZ: libc::size_t = 131072;
    // FreeBSD's is actually arch-dependent, but never more than 40960.
    // No harm in being generous.
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub const SIGSTKSZ: libc::size_t = 40960;

    pub const SIG_SETMASK: libc::c_int = 3;

    #[cfg(any(target_os = "macos",
              target_os = "ios"))]
    pub type sigset_t = u32;
    #[cfg(any(target_os = "freebsd", target_os = "dragonfly", target_os = "netbsd"))]
    #[repr(C)]
    pub struct sigset_t {
        bits: [u32; 4],
    }
    #[cfg(any(target_os = "bitrig", target_os = "openbsd"))]
    pub type sigset_t = libc::c_uint;

    // This structure has more fields, but we're not all that interested in
    // them.
    #[cfg(any(target_os = "macos", target_os = "ios",
              target_os = "freebsd", target_os = "dragonfly"))]
    #[repr(C)]
    pub struct siginfo {
        pub _signo: libc::c_int,
        pub _errno: libc::c_int,
        pub _code: libc::c_int,
        pub _pid: libc::pid_t,
        pub _uid: libc::uid_t,
        pub _status: libc::c_int,
        pub si_addr: *mut libc::c_void
    }
    #[cfg(any(target_os = "bitrig", target_os = "netbsd", target_os = "openbsd"))]
    #[repr(C)]
    pub struct siginfo {
        pub si_signo: libc::c_int,
        pub si_code: libc::c_int,
        pub si_errno: libc::c_int,
        pub si_addr: *mut libc::c_void
    }

    #[cfg(any(target_os = "macos", target_os = "ios",
              target_os = "bitrig", target_os = "netbsd", target_os = "openbsd"))]
    #[repr(C)]
    pub struct sigaction {
        pub sa_sigaction: sighandler_t,
        pub sa_mask: sigset_t,
        pub sa_flags: libc::c_int,
    }

    #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
    #[repr(C)]
    pub struct sigaction {
        pub sa_sigaction: sighandler_t,
        pub sa_flags: libc::c_int,
        pub sa_mask: sigset_t,
    }

    #[repr(C)]
    pub struct sigaltstack {
        pub ss_sp: *mut libc::c_void,
        pub ss_size: libc::size_t,
        pub ss_flags: libc::c_int,
    }
}
