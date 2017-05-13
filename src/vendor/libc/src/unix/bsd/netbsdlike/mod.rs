pub type time_t = i64;
pub type mode_t = u32;
pub type nlink_t = ::uint32_t;
pub type ino_t = ::uint64_t;
pub type pthread_key_t = ::c_int;
pub type rlim_t = u64;
pub type speed_t = ::c_uint;
pub type tcflag_t = ::c_uint;
pub type nl_item = c_long;
pub type clockid_t = ::c_int;
pub type id_t = ::uint32_t;
pub type sem_t = *mut sem;

pub enum timezone {}
pub enum sem {}

s! {
    pub struct sigaction {
        pub sa_sigaction: ::sighandler_t,
        pub sa_mask: ::sigset_t,
        pub sa_flags: ::c_int,
    }

    pub struct stack_t {
        pub ss_sp: *mut ::c_void,
        pub ss_size: ::size_t,
        pub ss_flags: ::c_int,
    }

    pub struct sockaddr_in {
        pub sin_len: u8,
        pub sin_family: ::sa_family_t,
        pub sin_port: ::in_port_t,
        pub sin_addr: ::in_addr,
        pub sin_zero: [::int8_t; 8],
    }

    pub struct termios {
        pub c_iflag: ::tcflag_t,
        pub c_oflag: ::tcflag_t,
        pub c_cflag: ::tcflag_t,
        pub c_lflag: ::tcflag_t,
        pub c_cc: [::cc_t; ::NCCS],
        pub c_ispeed: ::c_int,
        pub c_ospeed: ::c_int,
    }

    pub struct flock {
        pub l_start: ::off_t,
        pub l_len: ::off_t,
        pub l_pid: ::pid_t,
        pub l_type: ::c_short,
        pub l_whence: ::c_short,
    }
}

pub const D_T_FMT: ::nl_item = 0;
pub const D_FMT: ::nl_item = 1;
pub const T_FMT: ::nl_item = 2;
pub const T_FMT_AMPM: ::nl_item = 3;
pub const AM_STR: ::nl_item = 4;
pub const PM_STR: ::nl_item = 5;

pub const DAY_1: ::nl_item = 6;
pub const DAY_2: ::nl_item = 7;
pub const DAY_3: ::nl_item = 8;
pub const DAY_4: ::nl_item = 9;
pub const DAY_5: ::nl_item = 10;
pub const DAY_6: ::nl_item = 11;
pub const DAY_7: ::nl_item = 12;

pub const ABDAY_1: ::nl_item = 13;
pub const ABDAY_2: ::nl_item = 14;
pub const ABDAY_3: ::nl_item = 15;
pub const ABDAY_4: ::nl_item = 16;
pub const ABDAY_5: ::nl_item = 17;
pub const ABDAY_6: ::nl_item = 18;
pub const ABDAY_7: ::nl_item = 19;

pub const MON_1: ::nl_item = 20;
pub const MON_2: ::nl_item = 21;
pub const MON_3: ::nl_item = 22;
pub const MON_4: ::nl_item = 23;
pub const MON_5: ::nl_item = 24;
pub const MON_6: ::nl_item = 25;
pub const MON_7: ::nl_item = 26;
pub const MON_8: ::nl_item = 27;
pub const MON_9: ::nl_item = 28;
pub const MON_10: ::nl_item = 29;
pub const MON_11: ::nl_item = 30;
pub const MON_12: ::nl_item = 31;

pub const ABMON_1: ::nl_item = 32;
pub const ABMON_2: ::nl_item = 33;
pub const ABMON_3: ::nl_item = 34;
pub const ABMON_4: ::nl_item = 35;
pub const ABMON_5: ::nl_item = 36;
pub const ABMON_6: ::nl_item = 37;
pub const ABMON_7: ::nl_item = 38;
pub const ABMON_8: ::nl_item = 39;
pub const ABMON_9: ::nl_item = 40;
pub const ABMON_10: ::nl_item = 41;
pub const ABMON_11: ::nl_item = 42;
pub const ABMON_12: ::nl_item = 43;

pub const RADIXCHAR: ::nl_item = 44;
pub const THOUSEP: ::nl_item = 45;
pub const YESSTR: ::nl_item = 46;
pub const YESEXPR: ::nl_item = 47;
pub const NOSTR: ::nl_item = 48;
pub const NOEXPR: ::nl_item = 49;
pub const CRNCYSTR: ::nl_item = 50;

pub const CODESET: ::nl_item = 51;

pub const EXIT_FAILURE : ::c_int = 1;
pub const EXIT_SUCCESS : ::c_int = 0;
pub const RAND_MAX : ::c_int = 2147483647;
pub const EOF : ::c_int = -1;
pub const SEEK_SET : ::c_int = 0;
pub const SEEK_CUR : ::c_int = 1;
pub const SEEK_END : ::c_int = 2;
pub const _IOFBF : ::c_int = 0;
pub const _IONBF : ::c_int = 2;
pub const _IOLBF : ::c_int = 1;
pub const BUFSIZ : ::c_uint = 1024;
pub const FOPEN_MAX : ::c_uint = 20;
pub const FILENAME_MAX : ::c_uint = 1024;
pub const L_tmpnam : ::c_uint = 1024;
pub const O_RDONLY : ::c_int = 0;
pub const O_WRONLY : ::c_int = 1;
pub const O_RDWR : ::c_int = 2;
pub const O_ACCMODE : ::c_int = 3;
pub const O_APPEND : ::c_int = 8;
pub const O_CREAT : ::c_int = 512;
pub const O_EXCL : ::c_int = 2048;
pub const O_NOCTTY : ::c_int = 32768;
pub const O_TRUNC : ::c_int = 1024;
pub const O_SYNC : ::c_int = 128;
pub const S_IFIFO : mode_t = 4096;
pub const S_IFCHR : mode_t = 8192;
pub const S_IFBLK : mode_t = 24576;
pub const S_IFDIR : mode_t = 16384;
pub const S_IFREG : mode_t = 32768;
pub const S_IFLNK : mode_t = 40960;
pub const S_IFSOCK : mode_t = 49152;
pub const S_IFMT : mode_t = 61440;
pub const S_IEXEC : mode_t = 64;
pub const S_IWRITE : mode_t = 128;
pub const S_IREAD : mode_t = 256;
pub const S_IRWXU : mode_t = 448;
pub const S_IXUSR : mode_t = 64;
pub const S_IWUSR : mode_t = 128;
pub const S_IRUSR : mode_t = 256;
pub const S_IRWXG : mode_t = 56;
pub const S_IXGRP : mode_t = 8;
pub const S_IWGRP : mode_t = 16;
pub const S_IRGRP : mode_t = 32;
pub const S_IRWXO : mode_t = 7;
pub const S_IXOTH : mode_t = 1;
pub const S_IWOTH : mode_t = 2;
pub const S_IROTH : mode_t = 4;
pub const F_OK : ::c_int = 0;
pub const R_OK : ::c_int = 4;
pub const W_OK : ::c_int = 2;
pub const X_OK : ::c_int = 1;
pub const STDIN_FILENO : ::c_int = 0;
pub const STDOUT_FILENO : ::c_int = 1;
pub const STDERR_FILENO : ::c_int = 2;
pub const F_LOCK : ::c_int = 1;
pub const F_TEST : ::c_int = 3;
pub const F_TLOCK : ::c_int = 2;
pub const F_ULOCK : ::c_int = 0;
pub const F_GETLK: ::c_int = 7;
pub const F_SETLK: ::c_int = 8;
pub const F_SETLKW: ::c_int = 9;
pub const SIGHUP : ::c_int = 1;
pub const SIGINT : ::c_int = 2;
pub const SIGQUIT : ::c_int = 3;
pub const SIGILL : ::c_int = 4;
pub const SIGABRT : ::c_int = 6;
pub const SIGEMT: ::c_int = 7;
pub const SIGFPE : ::c_int = 8;
pub const SIGKILL : ::c_int = 9;
pub const SIGSEGV : ::c_int = 11;
pub const SIGPIPE : ::c_int = 13;
pub const SIGALRM : ::c_int = 14;
pub const SIGTERM : ::c_int = 15;
pub const SIGSTKSZ : ::size_t = 40960;

pub const PROT_NONE : ::c_int = 0;
pub const PROT_READ : ::c_int = 1;
pub const PROT_WRITE : ::c_int = 2;
pub const PROT_EXEC : ::c_int = 4;

pub const MAP_FILE : ::c_int = 0x0000;
pub const MAP_SHARED : ::c_int = 0x0001;
pub const MAP_PRIVATE : ::c_int = 0x0002;
pub const MAP_FIXED : ::c_int = 0x0010;
pub const MAP_ANON : ::c_int = 0x1000;

pub const MAP_FAILED : *mut ::c_void = !0 as *mut ::c_void;

pub const MCL_CURRENT : ::c_int = 0x0001;
pub const MCL_FUTURE : ::c_int = 0x0002;

pub const MS_ASYNC : ::c_int = 0x0001;

pub const EPERM : ::c_int = 1;
pub const ENOENT : ::c_int = 2;
pub const ESRCH : ::c_int = 3;
pub const EINTR : ::c_int = 4;
pub const EIO : ::c_int = 5;
pub const ENXIO : ::c_int = 6;
pub const E2BIG : ::c_int = 7;
pub const ENOEXEC : ::c_int = 8;
pub const EBADF : ::c_int = 9;
pub const ECHILD : ::c_int = 10;
pub const EDEADLK : ::c_int = 11;
pub const ENOMEM : ::c_int = 12;
pub const EACCES : ::c_int = 13;
pub const EFAULT : ::c_int = 14;
pub const ENOTBLK : ::c_int = 15;
pub const EBUSY : ::c_int = 16;
pub const EEXIST : ::c_int = 17;
pub const EXDEV : ::c_int = 18;
pub const ENODEV : ::c_int = 19;
pub const ENOTDIR : ::c_int = 20;
pub const EISDIR : ::c_int = 21;
pub const EINVAL : ::c_int = 22;
pub const ENFILE : ::c_int = 23;
pub const EMFILE : ::c_int = 24;
pub const ENOTTY : ::c_int = 25;
pub const ETXTBSY : ::c_int = 26;
pub const EFBIG : ::c_int = 27;
pub const ENOSPC : ::c_int = 28;
pub const ESPIPE : ::c_int = 29;
pub const EROFS : ::c_int = 30;
pub const EMLINK : ::c_int = 31;
pub const EPIPE : ::c_int = 32;
pub const EDOM : ::c_int = 33;
pub const ERANGE : ::c_int = 34;
pub const EAGAIN : ::c_int = 35;
pub const EWOULDBLOCK : ::c_int = 35;
pub const EINPROGRESS : ::c_int = 36;
pub const EALREADY : ::c_int = 37;
pub const ENOTSOCK : ::c_int = 38;
pub const EDESTADDRREQ : ::c_int = 39;
pub const EMSGSIZE : ::c_int = 40;
pub const EPROTOTYPE : ::c_int = 41;
pub const ENOPROTOOPT : ::c_int = 42;
pub const EPROTONOSUPPORT : ::c_int = 43;
pub const ESOCKTNOSUPPORT : ::c_int = 44;
pub const EOPNOTSUPP : ::c_int = 45;
pub const EPFNOSUPPORT : ::c_int = 46;
pub const EAFNOSUPPORT : ::c_int = 47;
pub const EADDRINUSE : ::c_int = 48;
pub const EADDRNOTAVAIL : ::c_int = 49;
pub const ENETDOWN : ::c_int = 50;
pub const ENETUNREACH : ::c_int = 51;
pub const ENETRESET : ::c_int = 52;
pub const ECONNABORTED : ::c_int = 53;
pub const ECONNRESET : ::c_int = 54;
pub const ENOBUFS : ::c_int = 55;
pub const EISCONN : ::c_int = 56;
pub const ENOTCONN : ::c_int = 57;
pub const ESHUTDOWN : ::c_int = 58;
pub const ETOOMANYREFS : ::c_int = 59;
pub const ETIMEDOUT : ::c_int = 60;
pub const ECONNREFUSED : ::c_int = 61;
pub const ELOOP : ::c_int = 62;
pub const ENAMETOOLONG : ::c_int = 63;
pub const EHOSTDOWN : ::c_int = 64;
pub const EHOSTUNREACH : ::c_int = 65;
pub const ENOTEMPTY : ::c_int = 66;
pub const EPROCLIM : ::c_int = 67;
pub const EUSERS : ::c_int = 68;
pub const EDQUOT : ::c_int = 69;
pub const ESTALE : ::c_int = 70;
pub const EREMOTE : ::c_int = 71;
pub const EBADRPC : ::c_int = 72;
pub const ERPCMISMATCH : ::c_int = 73;
pub const EPROGUNAVAIL : ::c_int = 74;
pub const EPROGMISMATCH : ::c_int = 75;
pub const EPROCUNAVAIL : ::c_int = 76;
pub const ENOLCK : ::c_int = 77;
pub const ENOSYS : ::c_int = 78;
pub const EFTYPE : ::c_int = 79;
pub const EAUTH : ::c_int = 80;
pub const ENEEDAUTH : ::c_int = 81;

pub const F_DUPFD : ::c_int = 0;
pub const F_GETFD : ::c_int = 1;
pub const F_SETFD : ::c_int = 2;
pub const F_GETFL : ::c_int = 3;
pub const F_SETFL : ::c_int = 4;

pub const SIGTRAP : ::c_int = 5;

pub const GLOB_APPEND   : ::c_int = 0x0001;
pub const GLOB_DOOFFS   : ::c_int = 0x0002;
pub const GLOB_ERR      : ::c_int = 0x0004;
pub const GLOB_MARK     : ::c_int = 0x0008;
pub const GLOB_NOCHECK  : ::c_int = 0x0010;
pub const GLOB_NOSORT   : ::c_int = 0x0020;
pub const GLOB_NOESCAPE : ::c_int = 0x1000;

pub const GLOB_NOSPACE  : ::c_int = -1;
pub const GLOB_ABORTED  : ::c_int = -2;
pub const GLOB_NOMATCH  : ::c_int = -3;
pub const GLOB_NOSYS : ::c_int = -4;

pub const POSIX_MADV_NORMAL : ::c_int = 0;
pub const POSIX_MADV_RANDOM : ::c_int = 1;
pub const POSIX_MADV_SEQUENTIAL : ::c_int = 2;
pub const POSIX_MADV_WILLNEED : ::c_int = 3;
pub const POSIX_MADV_DONTNEED : ::c_int = 4;

pub const _SC_XOPEN_SHM : ::c_int = 30;

pub const PTHREAD_CREATE_JOINABLE : ::c_int = 0;
pub const PTHREAD_CREATE_DETACHED : ::c_int = 1;

// http://man.openbsd.org/OpenBSD-current/man2/clock_getres.2
// The man page says clock_gettime(3) can accept various values as clockid_t but
// http://fxr.watson.org/fxr/source/kern/kern_time.c?v=OPENBSD;im=excerpts#L161
// the implementation rejects anything other than the below two
//
// http://netbsd.gw.com/cgi-bin/man-cgi?clock_gettime
// https://github.com/jsonn/src/blob/HEAD/sys/kern/subr_time.c#L222
// Basically the same goes for NetBSD
pub const CLOCK_REALTIME: clockid_t = 0;
pub const CLOCK_MONOTONIC: clockid_t = 3;

pub const RLIMIT_CPU: ::c_int = 0;
pub const RLIMIT_FSIZE: ::c_int = 1;
pub const RLIMIT_DATA: ::c_int = 2;
pub const RLIMIT_STACK: ::c_int = 3;
pub const RLIMIT_CORE: ::c_int = 4;
pub const RLIMIT_RSS: ::c_int = 5;
pub const RLIMIT_MEMLOCK: ::c_int = 6;
pub const RLIMIT_NPROC: ::c_int = 7;
pub const RLIMIT_NOFILE: ::c_int = 8;

pub const RLIM_INFINITY: rlim_t = 0x7fff_ffff_ffff_ffff;
pub const RLIM_SAVED_MAX: rlim_t = RLIM_INFINITY;
pub const RLIM_SAVED_CUR: rlim_t = RLIM_INFINITY;

pub const RUSAGE_SELF: ::c_int = 0;
pub const RUSAGE_CHILDREN: ::c_int = -1;

pub const MADV_NORMAL : ::c_int = 0;
pub const MADV_RANDOM : ::c_int = 1;
pub const MADV_SEQUENTIAL : ::c_int = 2;
pub const MADV_WILLNEED : ::c_int = 3;
pub const MADV_DONTNEED : ::c_int = 4;
pub const MADV_FREE : ::c_int = 6;

pub const AF_UNIX: ::c_int = 1;
pub const AF_INET: ::c_int = 2;
pub const AF_INET6: ::c_int = 24;
pub const SOCK_STREAM: ::c_int = 1;
pub const SOCK_DGRAM: ::c_int = 2;
pub const SOCK_RAW: ::c_int = 3;
pub const SOCK_SEQPACKET: ::c_int = 5;
pub const IPPROTO_TCP: ::c_int = 6;
pub const IPPROTO_IP: ::c_int = 0;
pub const IPPROTO_IPV6: ::c_int = 41;
pub const IP_MULTICAST_TTL: ::c_int = 10;
pub const IP_MULTICAST_LOOP: ::c_int = 11;
pub const IP_TTL: ::c_int = 4;
pub const IP_HDRINCL: ::c_int = 2;
pub const IP_ADD_MEMBERSHIP: ::c_int = 12;
pub const IP_DROP_MEMBERSHIP: ::c_int = 13;

pub const TCP_NODELAY: ::c_int = 0x01;
pub const SOL_SOCKET: ::c_int = 0xffff;
pub const SO_DEBUG: ::c_int = 0x01;
pub const SO_ACCEPTCONN: ::c_int = 0x0002;
pub const SO_REUSEADDR: ::c_int = 0x0004;
pub const SO_KEEPALIVE: ::c_int = 0x0008;
pub const SO_DONTROUTE: ::c_int = 0x0010;
pub const SO_BROADCAST: ::c_int = 0x0020;
pub const SO_USELOOPBACK: ::c_int = 0x0040;
pub const SO_LINGER: ::c_int = 0x0080;
pub const SO_OOBINLINE: ::c_int = 0x0100;
pub const SO_REUSEPORT: ::c_int = 0x0200;
pub const SO_SNDBUF: ::c_int = 0x1001;
pub const SO_RCVBUF: ::c_int = 0x1002;
pub const SO_SNDLOWAT: ::c_int = 0x1003;
pub const SO_RCVLOWAT: ::c_int = 0x1004;
pub const SO_ERROR: ::c_int = 0x1007;
pub const SO_TYPE: ::c_int = 0x1008;

pub const MSG_NOSIGNAL: ::c_int = 0x400;

pub const IFF_LOOPBACK: ::c_int = 0x8;

pub const SHUT_RD: ::c_int = 0;
pub const SHUT_WR: ::c_int = 1;
pub const SHUT_RDWR: ::c_int = 2;

pub const LOCK_SH: ::c_int = 1;
pub const LOCK_EX: ::c_int = 2;
pub const LOCK_NB: ::c_int = 4;
pub const LOCK_UN: ::c_int = 8;

pub const O_NONBLOCK : ::c_int = 4;

pub const IPPROTO_RAW : ::c_int = 255;

pub const _SC_ARG_MAX : ::c_int = 1;
pub const _SC_CHILD_MAX : ::c_int = 2;
pub const _SC_NGROUPS_MAX : ::c_int = 4;
pub const _SC_OPEN_MAX : ::c_int = 5;
pub const _SC_JOB_CONTROL : ::c_int = 6;
pub const _SC_SAVED_IDS : ::c_int = 7;
pub const _SC_VERSION : ::c_int = 8;
pub const _SC_BC_BASE_MAX : ::c_int = 9;
pub const _SC_BC_DIM_MAX : ::c_int = 10;
pub const _SC_BC_SCALE_MAX : ::c_int = 11;
pub const _SC_BC_STRING_MAX : ::c_int = 12;
pub const _SC_COLL_WEIGHTS_MAX : ::c_int = 13;
pub const _SC_EXPR_NEST_MAX : ::c_int = 14;
pub const _SC_LINE_MAX : ::c_int = 15;
pub const _SC_RE_DUP_MAX : ::c_int = 16;
pub const _SC_2_VERSION : ::c_int = 17;
pub const _SC_2_C_BIND : ::c_int = 18;
pub const _SC_2_C_DEV : ::c_int = 19;
pub const _SC_2_CHAR_TERM : ::c_int = 20;
pub const _SC_2_FORT_DEV : ::c_int = 21;
pub const _SC_2_FORT_RUN : ::c_int = 22;
pub const _SC_2_LOCALEDEF : ::c_int = 23;
pub const _SC_2_SW_DEV : ::c_int = 24;
pub const _SC_2_UPE : ::c_int = 25;
pub const _SC_STREAM_MAX : ::c_int = 26;
pub const _SC_TZNAME_MAX : ::c_int = 27;
pub const _SC_PAGESIZE : ::c_int = 28;
pub const _SC_PAGE_SIZE: ::c_int = _SC_PAGESIZE;
pub const _SC_FSYNC : ::c_int = 29;

pub const Q_GETQUOTA: ::c_int = 0x300;
pub const Q_SETQUOTA: ::c_int = 0x400;

pub const RTLD_GLOBAL: ::c_int = 0x100;

pub const LOG_NFACILITIES: ::c_int = 24;

pub const HW_NCPU: ::c_int = 3;

pub const SEM_FAILED: *mut sem_t = 0 as *mut sem_t;

f! {
    pub fn WSTOPSIG(status: ::c_int) -> ::c_int {
        status >> 8
    }

    pub fn WIFSIGNALED(status: ::c_int) -> bool {
        (status & 0o177) != 0o177 && (status & 0o177) != 0
    }

    pub fn WIFSTOPPED(status: ::c_int) -> bool {
        (status & 0o177) == 0o177
    }
}

#[link(name = "util")]
extern {
    pub fn mincore(addr: *mut ::c_void, len: ::size_t,
                   vec: *mut ::c_char) -> ::c_int;
    #[cfg_attr(target_os = "netbsd", link_name = "__clock_getres50")]
    pub fn clock_getres(clk_id: clockid_t, tp: *mut ::timespec) -> ::c_int;
    #[cfg_attr(target_os = "netbsd", link_name = "__clock_gettime50")]
    pub fn clock_gettime(clk_id: clockid_t, tp: *mut ::timespec) -> ::c_int;
    pub fn __errno() -> *mut ::c_int;
    pub fn shm_open(name: *const ::c_char, oflag: ::c_int, mode: ::mode_t)
                    -> ::c_int;
    pub fn memrchr(cx: *const ::c_void,
                   c: ::c_int,
                   n: ::size_t) -> *mut ::c_void;
    pub fn mkostemp(template: *mut ::c_char, flags: ::c_int) -> ::c_int;
    pub fn mkostemps(template: *mut ::c_char,
                     suffixlen: ::c_int,
                     flags: ::c_int) -> ::c_int;
    pub fn futimens(fd: ::c_int, times: *const ::timespec) -> ::c_int;
    pub fn fdatasync(fd: ::c_int) -> ::c_int;
    pub fn openpty(amaster: *mut ::c_int,
                   aslave: *mut ::c_int,
                   name: *mut ::c_char,
                   termp: *mut termios,
                   winp: *mut ::winsize) -> ::c_int;
    pub fn forkpty(amaster: *mut ::c_int,
                   name: *mut ::c_char,
                   termp: *mut termios,
                   winp: *mut ::winsize) -> ::pid_t;
    pub fn getpriority(which: ::c_int, who: ::id_t) -> ::c_int;
    pub fn setpriority(which: ::c_int, who: ::id_t, prio: ::c_int) -> ::c_int;

    pub fn openat(dirfd: ::c_int, pathname: *const ::c_char,
                  flags: ::c_int, ...) -> ::c_int;
    pub fn faccessat(dirfd: ::c_int, pathname: *const ::c_char,
                     mode: ::c_int, flags: ::c_int) -> ::c_int;
    pub fn fchmodat(dirfd: ::c_int, pathname: *const ::c_char,
                    mode: ::mode_t, flags: ::c_int) -> ::c_int;
    pub fn fchownat(dirfd: ::c_int, pathname: *const ::c_char,
                    owner: ::uid_t, group: ::gid_t,
                    flags: ::c_int) -> ::c_int;
    pub fn fstatat(dirfd: ::c_int, pathname: *const ::c_char,
                   buf: *mut stat, flags: ::c_int) -> ::c_int;
    pub fn linkat(olddirfd: ::c_int, oldpath: *const ::c_char,
                  newdirfd: ::c_int, newpath: *const ::c_char,
                  flags: ::c_int) -> ::c_int;
   pub fn mkdirat(dirfd: ::c_int, pathname: *const ::c_char,
                  mode: ::mode_t) -> ::c_int;
   pub fn mknodat(dirfd: ::c_int, pathname: *const ::c_char,
                 mode: ::mode_t, dev: dev_t) -> ::c_int;
   pub fn readlinkat(dirfd: ::c_int, pathname: *const ::c_char,
                     buf: *mut ::c_char, bufsiz: ::size_t) -> ::ssize_t;
   pub fn renameat(olddirfd: ::c_int, oldpath: *const ::c_char,
                   newdirfd: ::c_int, newpath: *const ::c_char)
                   -> ::c_int;
   pub fn symlinkat(target: *const ::c_char, newdirfd: ::c_int,
                    linkpath: *const ::c_char) -> ::c_int;
   pub fn unlinkat(dirfd: ::c_int, pathname: *const ::c_char,
                   flags: ::c_int) -> ::c_int;
   pub fn mkfifoat(dirfd: ::c_int, pathname: *const ::c_char,
                   mode: ::mode_t) -> ::c_int;
   pub fn sem_timedwait(sem: *mut sem_t,
                        abstime: *const ::timespec) -> ::c_int;
   pub fn pthread_condattr_setclock(attr: *mut pthread_condattr_t,
                                    clock_id: clockid_t) -> ::c_int;
    pub fn sethostname(name: *const ::c_char, len: ::size_t) -> ::c_int;
    pub fn pthread_mutex_timedlock(lock: *mut pthread_mutex_t,
                                   abstime: *const ::timespec) -> ::c_int;
}

cfg_if! {
    if #[cfg(target_os = "netbsd")] {
        mod netbsd;
        pub use self::netbsd::*;
    } else if #[cfg(any(target_os = "openbsd", target_os = "bitrig"))] {
        mod openbsdlike;
        pub use self::openbsdlike::*;
    } else {
        // Unknown target_os
    }
}
