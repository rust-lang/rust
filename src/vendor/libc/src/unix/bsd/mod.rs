use dox::mem;

pub type c_char = i8;
pub type wchar_t = i32;
pub type off_t = i64;
pub type useconds_t = u32;
pub type blkcnt_t = i64;
pub type socklen_t = u32;
pub type sa_family_t = u8;
pub type pthread_t = ::uintptr_t;
pub type nfds_t = ::c_uint;

s! {
    pub struct sockaddr {
        pub sa_len: u8,
        pub sa_family: sa_family_t,
        pub sa_data: [::c_char; 14],
    }

    pub struct sockaddr_in6 {
        pub sin6_len: u8,
        pub sin6_family: sa_family_t,
        pub sin6_port: ::in_port_t,
        pub sin6_flowinfo: u32,
        pub sin6_addr: ::in6_addr,
        pub sin6_scope_id: u32,
    }

    pub struct sockaddr_un {
        pub sun_len: u8,
        pub sun_family: sa_family_t,
        pub sun_path: [c_char; 104]
    }

    pub struct passwd {
        pub pw_name: *mut ::c_char,
        pub pw_passwd: *mut ::c_char,
        pub pw_uid: ::uid_t,
        pub pw_gid: ::gid_t,
        pub pw_change: ::time_t,
        pub pw_class: *mut ::c_char,
        pub pw_gecos: *mut ::c_char,
        pub pw_dir: *mut ::c_char,
        pub pw_shell: *mut ::c_char,
        pub pw_expire: ::time_t,

        #[cfg(not(any(target_os = "macos",
                      target_os = "ios",
                      target_os = "netbsd",
                      target_os = "openbsd")))]
        pub pw_fields: ::c_int,
    }

    pub struct ifaddrs {
        pub ifa_next: *mut ifaddrs,
        pub ifa_name: *mut ::c_char,
        pub ifa_flags: ::c_uint,
        pub ifa_addr: *mut ::sockaddr,
        pub ifa_netmask: *mut ::sockaddr,
        pub ifa_dstaddr: *mut ::sockaddr,
        pub ifa_data: *mut ::c_void
    }

    pub struct fd_set {
        #[cfg(all(target_pointer_width = "64",
                  any(target_os = "freebsd", target_os = "dragonfly")))]
        fds_bits: [i64; FD_SETSIZE / 64],
        #[cfg(not(all(target_pointer_width = "64",
                      any(target_os = "freebsd", target_os = "dragonfly"))))]
        fds_bits: [i32; FD_SETSIZE / 32],
    }

    pub struct tm {
        pub tm_sec: ::c_int,
        pub tm_min: ::c_int,
        pub tm_hour: ::c_int,
        pub tm_mday: ::c_int,
        pub tm_mon: ::c_int,
        pub tm_year: ::c_int,
        pub tm_wday: ::c_int,
        pub tm_yday: ::c_int,
        pub tm_isdst: ::c_int,
        pub tm_gmtoff: ::c_long,
        pub tm_zone: *mut ::c_char,
    }

    pub struct utsname {
        #[cfg(not(target_os = "dragonfly"))]
        pub sysname: [::c_char; 256],
        #[cfg(target_os = "dragonfly")]
        pub sysname: [::c_char; 32],
        #[cfg(not(target_os = "dragonfly"))]
        pub nodename: [::c_char; 256],
        #[cfg(target_os = "dragonfly")]
        pub nodename: [::c_char; 32],
        #[cfg(not(target_os = "dragonfly"))]
        pub release: [::c_char; 256],
        #[cfg(target_os = "dragonfly")]
        pub release: [::c_char; 32],
        #[cfg(not(target_os = "dragonfly"))]
        pub version: [::c_char; 256],
        #[cfg(target_os = "dragonfly")]
        pub version: [::c_char; 32],
        #[cfg(not(target_os = "dragonfly"))]
        pub machine: [::c_char; 256],
        #[cfg(target_os = "dragonfly")]
        pub machine: [::c_char; 32],
    }

    pub struct msghdr {
        pub msg_name: *mut ::c_void,
        pub msg_namelen: ::socklen_t,
        pub msg_iov: *mut ::iovec,
        pub msg_iovlen: ::c_int,
        pub msg_control: *mut ::c_void,
        pub msg_controllen: ::socklen_t,
        pub msg_flags: ::c_int,
    }

    pub struct fsid_t {
        __fsid_val: [::int32_t; 2],
    }

    pub struct if_nameindex {
        pub if_index: ::c_uint,
        pub if_name: *mut ::c_char,
    }
}

pub const LC_ALL: ::c_int = 0;
pub const LC_COLLATE: ::c_int = 1;
pub const LC_CTYPE: ::c_int = 2;
pub const LC_MONETARY: ::c_int = 3;
pub const LC_NUMERIC: ::c_int = 4;
pub const LC_TIME: ::c_int = 5;
pub const LC_MESSAGES: ::c_int = 6;

pub const FIOCLEX: ::c_ulong = 0x20006601;
pub const FIONBIO: ::c_ulong = 0x8004667e;

pub const PATH_MAX: ::c_int = 1024;

pub const SA_ONSTACK: ::c_int = 0x0001;
pub const SA_SIGINFO: ::c_int = 0x0040;
pub const SA_RESTART: ::c_int = 0x0002;
pub const SA_RESETHAND: ::c_int = 0x0004;
pub const SA_NOCLDSTOP: ::c_int = 0x0008;
pub const SA_NODEFER: ::c_int = 0x0010;
pub const SA_NOCLDWAIT: ::c_int = 0x0020;

pub const SS_ONSTACK: ::c_int = 1;
pub const SS_DISABLE: ::c_int = 4;

pub const SIGCHLD: ::c_int = 20;
pub const SIGBUS: ::c_int = 10;
pub const SIGUSR1: ::c_int = 30;
pub const SIGUSR2: ::c_int = 31;
pub const SIGCONT: ::c_int = 19;
pub const SIGSTOP: ::c_int = 17;
pub const SIGTSTP: ::c_int = 18;
pub const SIGURG: ::c_int = 16;
pub const SIGIO: ::c_int = 23;
pub const SIGSYS: ::c_int = 12;
pub const SIGTTIN: ::c_int = 21;
pub const SIGTTOU: ::c_int = 22;
pub const SIGXCPU: ::c_int = 24;
pub const SIGXFSZ: ::c_int = 25;
pub const SIGVTALRM: ::c_int = 26;
pub const SIGPROF: ::c_int = 27;
pub const SIGWINCH: ::c_int = 28;
pub const SIGINFO: ::c_int = 29;

pub const SIG_SETMASK: ::c_int = 3;
pub const SIG_BLOCK: ::c_int = 0x1;
pub const SIG_UNBLOCK: ::c_int = 0x2;

pub const IPV6_MULTICAST_LOOP: ::c_int = 11;
pub const IPV6_V6ONLY: ::c_int = 27;

pub const ST_RDONLY: ::c_ulong = 1;

pub const NCCS: usize = 20;

pub const O_ASYNC: ::c_int = 0x40;
pub const O_FSYNC: ::c_int = 0x80;
pub const O_NDELAY: ::c_int = 0x4;
pub const O_NOFOLLOW: ::c_int = 0x100;

pub const F_GETOWN: ::c_int = 5;
pub const F_SETOWN: ::c_int = 6;

pub const MNT_FORCE: ::c_int = 0x80000;

pub const Q_SYNC: ::c_int = 0x600;
pub const Q_QUOTAON: ::c_int = 0x100;
pub const Q_QUOTAOFF: ::c_int = 0x200;

pub const TCIOFF: ::c_int = 3;
pub const TCION: ::c_int = 4;
pub const TCOOFF: ::c_int = 1;
pub const TCOON: ::c_int = 2;
pub const TCIFLUSH: ::c_int = 1;
pub const TCOFLUSH: ::c_int = 2;
pub const TCIOFLUSH: ::c_int = 3;
pub const TCSANOW: ::c_int = 0;
pub const TCSADRAIN: ::c_int = 1;
pub const TCSAFLUSH: ::c_int = 2;
pub const VEOF: usize = 0;
pub const VEOL: usize = 1;
pub const VEOL2: usize = 2;
pub const VERASE: usize = 3;
pub const VWERASE: usize = 4;
pub const VKILL: usize = 5;
pub const VREPRINT: usize = 6;
pub const VINTR: usize = 8;
pub const VQUIT: usize = 9;
pub const VSUSP: usize = 10;
pub const VSTART: usize = 12;
pub const VSTOP: usize = 13;
pub const VLNEXT: usize = 14;
pub const VDISCARD: usize = 15;
pub const VMIN: usize = 16;
pub const VTIME: usize = 17;
pub const IGNBRK: ::tcflag_t = 0x00000001;
pub const BRKINT: ::tcflag_t = 0x00000002;
pub const IGNPAR: ::tcflag_t = 0x00000004;
pub const PARMRK: ::tcflag_t = 0x00000008;
pub const INPCK: ::tcflag_t = 0x00000010;
pub const ISTRIP: ::tcflag_t = 0x00000020;
pub const INLCR: ::tcflag_t = 0x00000040;
pub const IGNCR: ::tcflag_t = 0x00000080;
pub const ICRNL: ::tcflag_t = 0x00000100;
pub const IXON: ::tcflag_t = 0x00000200;
pub const IXOFF: ::tcflag_t = 0x00000400;
pub const IXANY: ::tcflag_t = 0x00000800;
pub const IMAXBEL: ::tcflag_t = 0x00002000;
pub const OPOST: ::tcflag_t = 0x1;
pub const ONLCR: ::tcflag_t = 0x2;
pub const CSIZE: ::tcflag_t = 0x00000300;
pub const CS5: ::tcflag_t = 0x00000000;
pub const CS6: ::tcflag_t = 0x00000100;
pub const CS7: ::tcflag_t = 0x00000200;
pub const CS8: ::tcflag_t = 0x00000300;
pub const CSTOPB: ::tcflag_t = 0x00000400;
pub const CREAD: ::tcflag_t = 0x00000800;
pub const PARENB: ::tcflag_t = 0x00001000;
pub const PARODD: ::tcflag_t = 0x00002000;
pub const HUPCL: ::tcflag_t = 0x00004000;
pub const CLOCAL: ::tcflag_t = 0x00008000;
pub const ECHOKE: ::tcflag_t = 0x00000001;
pub const ECHOE: ::tcflag_t = 0x00000002;
pub const ECHOK: ::tcflag_t = 0x00000004;
pub const ECHO: ::tcflag_t = 0x00000008;
pub const ECHONL: ::tcflag_t = 0x00000010;
pub const ECHOPRT: ::tcflag_t = 0x00000020;
pub const ECHOCTL: ::tcflag_t = 0x00000040;
pub const ISIG: ::tcflag_t = 0x00000080;
pub const ICANON: ::tcflag_t = 0x00000100;
pub const IEXTEN: ::tcflag_t = 0x00000400;
pub const EXTPROC: ::tcflag_t = 0x00000800;
pub const TOSTOP: ::tcflag_t = 0x00400000;
pub const FLUSHO: ::tcflag_t = 0x00800000;
pub const PENDIN: ::tcflag_t = 0x20000000;
pub const NOFLSH: ::tcflag_t = 0x80000000;

pub const WNOHANG: ::c_int = 0x00000001;
pub const WUNTRACED: ::c_int = 0x00000002;

pub const RTLD_NOW: ::c_int = 0x2;
pub const RTLD_DEFAULT: *mut ::c_void = -2isize as *mut ::c_void;

pub const LOG_CRON: ::c_int = 9 << 3;
pub const LOG_AUTHPRIV: ::c_int = 10 << 3;
pub const LOG_FTP: ::c_int = 11 << 3;
pub const LOG_PERROR: ::c_int = 0x20;

pub const PIPE_BUF: usize = 512;

f! {
    pub fn FD_CLR(fd: ::c_int, set: *mut fd_set) -> () {
        let bits = mem::size_of_val(&(*set).fds_bits[0]) * 8;
        let fd = fd as usize;
        (*set).fds_bits[fd / bits] &= !(1 << (fd % bits));
        return
    }

    pub fn FD_ISSET(fd: ::c_int, set: *mut fd_set) -> bool {
        let bits = mem::size_of_val(&(*set).fds_bits[0]) * 8;
        let fd = fd as usize;
        return ((*set).fds_bits[fd / bits] & (1 << (fd % bits))) != 0
    }

    pub fn FD_SET(fd: ::c_int, set: *mut fd_set) -> () {
        let bits = mem::size_of_val(&(*set).fds_bits[0]) * 8;
        let fd = fd as usize;
        (*set).fds_bits[fd / bits] |= 1 << (fd % bits);
        return
    }

    pub fn FD_ZERO(set: *mut fd_set) -> () {
        for slot in (*set).fds_bits.iter_mut() {
            *slot = 0;
        }
    }

    pub fn WTERMSIG(status: ::c_int) -> ::c_int {
        status & 0o177
    }

    pub fn WIFEXITED(status: ::c_int) -> bool {
        (status & 0o177) == 0
    }

    pub fn WEXITSTATUS(status: ::c_int) -> ::c_int {
        status >> 8
    }

    pub fn WCOREDUMP(status: ::c_int) -> bool {
        (status & 0o200) != 0
    }
}

extern {
    pub fn getifaddrs(ifap: *mut *mut ::ifaddrs) -> ::c_int;
    pub fn freeifaddrs(ifa: *mut ::ifaddrs);
    pub fn setgroups(ngroups: ::c_int,
                     ptr: *const ::gid_t) -> ::c_int;
    pub fn ioctl(fd: ::c_int, request: ::c_ulong, ...) -> ::c_int;
    pub fn kqueue() -> ::c_int;
    pub fn unmount(target: *const ::c_char, arg: ::c_int) -> ::c_int;
    pub fn syscall(num: ::c_int, ...) -> ::c_int;
    #[cfg_attr(target_os = "netbsd", link_name = "__getpwnam_r50")]
    pub fn getpwnam_r(name: *const ::c_char,
                      pwd: *mut passwd,
                      buf: *mut ::c_char,
                      buflen: ::size_t,
                      result: *mut *mut passwd) -> ::c_int;
    #[cfg_attr(target_os = "netbsd", link_name = "__getpwuid_r50")]
    pub fn getpwuid_r(uid: ::uid_t,
                      pwd: *mut passwd,
                      buf: *mut ::c_char,
                      buflen: ::size_t,
                      result: *mut *mut passwd) -> ::c_int;
    #[cfg_attr(target_os = "netbsd", link_name = "__getpwent50")]
    pub fn getpwent() -> *mut passwd;
    pub fn setpwent();
    pub fn getprogname() -> *const ::c_char;
    pub fn setprogname(name: *const ::c_char);
    pub fn getloadavg(loadavg: *mut ::c_double, nelem: ::c_int) -> ::c_int;
    pub fn if_nameindex() -> *mut if_nameindex;
    pub fn if_freenameindex(ptr: *mut if_nameindex);
}

cfg_if! {
    if #[cfg(any(target_os = "macos", target_os = "ios"))] {
        mod apple;
        pub use self::apple::*;
    } else if #[cfg(any(target_os = "openbsd", target_os = "netbsd",
                        target_os = "bitrig"))] {
        mod netbsdlike;
        pub use self::netbsdlike::*;
    } else if #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))] {
        mod freebsdlike;
        pub use self::freebsdlike::*;
    } else {
        // Unknown target_os
    }
}
