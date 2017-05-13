pub type c_char = i8;
pub type c_long = i64;
pub type c_ulong = u64;
pub type clockid_t = ::c_int;

pub type blkcnt_t = i64;
pub type clock_t = i64;
pub type daddr_t = i64;
pub type dev_t = u64;
pub type fsblkcnt_t = u64;
pub type fsfilcnt_t = u64;
pub type ino_t = i64;
pub type key_t = i32;
pub type major_t = u32;
pub type minor_t = u32;
pub type mode_t = u32;
pub type nlink_t = u32;
pub type rlim_t = u64;
pub type speed_t = u32;
pub type tcflag_t = u32;
pub type time_t = i64;
pub type wchar_t = i32;
pub type nfds_t = ::c_ulong;

pub type suseconds_t = ::c_long;
pub type off_t = i64;
pub type useconds_t = ::c_uint;
pub type socklen_t = u32;
pub type sa_family_t = u8;
pub type pthread_t = ::uintptr_t;
pub type pthread_key_t = ::c_uint;
pub type blksize_t = u32;
pub type fflags_t = u32;
pub type nl_item = ::c_int;

pub enum timezone {}

s! {
    pub struct sockaddr {
        pub sa_family: sa_family_t,
        pub sa_data: [::c_char; 14],
    }

    pub struct sockaddr_in {
        pub sin_family: sa_family_t,
        pub sin_port: ::in_port_t,
        pub sin_addr: ::in_addr,
        pub sin_zero: [::c_char; 8]
    }

    pub struct sockaddr_in6 {
        pub sin6_family: sa_family_t,
        pub sin6_port: ::in_port_t,
        pub sin6_flowinfo: u32,
        pub sin6_addr: ::in6_addr,
        pub sin6_scope_id: u32,
        pub __sin6_src_id: u32
    }

    pub struct sockaddr_un {
        pub sun_family: sa_family_t,
        pub sun_path: [c_char; 108]
    }

    pub struct passwd {
        pub pw_name: *mut ::c_char,
        pub pw_passwd: *mut ::c_char,
        pub pw_uid: ::uid_t,
        pub pw_gid: ::gid_t,
        pub pw_age: *mut ::c_char,
        pub pw_comment: *mut ::c_char,
        pub pw_gecos: *mut ::c_char,
        pub pw_dir: *mut ::c_char,
        pub pw_shell: *mut ::c_char
    }

    pub struct ifaddrs {
        pub ifa_next: *mut ifaddrs,
        pub ifa_name: *mut ::c_char,
        pub ifa_flags: ::c_ulong,
        pub ifa_addr: *mut ::sockaddr,
        pub ifa_netmask: *mut ::sockaddr,
        pub ifa_dstaddr: *mut ::sockaddr,
        pub ifa_data: *mut ::c_void
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
        pub tm_isdst: ::c_int
    }

    pub struct utsname {
        pub sysname: [::c_char; 257],
        pub nodename: [::c_char; 257],
        pub release: [::c_char; 257],
        pub version: [::c_char; 257],
        pub machine: [::c_char; 257],
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

    pub struct fd_set {
        fds_bits: [i32; FD_SETSIZE / 32],
    }

    pub struct pthread_attr_t {
        __pthread_attrp: *mut ::c_void
    }

    pub struct pthread_mutex_t {
        __pthread_mutex_flag1: u16,
        __pthread_mutex_flag2: u8,
        __pthread_mutex_ceiling: u8,
        __pthread_mutex_type: u16,
        __pthread_mutex_magic: u16,
        __pthread_mutex_lock: u64,
        __pthread_mutex_data: u64
    }

    pub struct pthread_mutexattr_t {
        __pthread_mutexattrp: *mut ::c_void
    }

    pub struct pthread_cond_t {
        __pthread_cond_flag: [u8; 4],
        __pthread_cond_type: u16,
        __pthread_cond_magic: u16,
        __pthread_cond_data: u64
    }

    pub struct pthread_condattr_t {
        __pthread_condattrp: *mut ::c_void,
    }

    pub struct pthread_rwlock_t {
        __pthread_rwlock_readers: i32,
        __pthread_rwlock_type: u16,
        __pthread_rwlock_magic: u16,
        __pthread_rwlock_mutex: ::pthread_mutex_t,
        __pthread_rwlock_readercv: ::pthread_cond_t,
        __pthread_rwlock_writercv: ::pthread_cond_t
    }

    pub struct dirent {
        pub d_ino: ::ino_t,
        pub d_off: ::off_t,
        pub d_reclen: u16,
        pub d_name: [::c_char; 1]
    }

    pub struct glob_t {
        pub gl_pathc: ::size_t,
        pub gl_pathv:  *mut *mut ::c_char,
        pub gl_offs: ::size_t,
        __unused1: *mut ::c_void,
        __unused2: ::c_int,
        __unused3: ::c_int,
        __unused4: ::c_int,
        __unused5: *mut ::c_void,
        __unused6: *mut ::c_void,
        __unused7: *mut ::c_void,
        __unused8: *mut ::c_void,
        __unused9: *mut ::c_void,
        __unused10: *mut ::c_void,
    }

    pub struct sockaddr_storage {
        pub ss_family: ::sa_family_t,
        __ss_pad1: [u8; 6],
        __ss_align: i64,
        __ss_pad2: [u8; 240],
    }

    pub struct addrinfo {
        pub ai_flags: ::c_int,
        pub ai_family: ::c_int,
        pub ai_socktype: ::c_int,
        pub ai_protocol: ::c_int,
        pub ai_addrlen: ::socklen_t,
        pub ai_canonname: *mut ::c_char,
        pub ai_addr: *mut ::sockaddr,
        pub ai_next: *mut addrinfo,
    }

    pub struct sigset_t {
        bits: [u32; 4],
    }

    pub struct siginfo_t {
        pub si_signo: ::c_int,
        pub si_code: ::c_int,
        pub si_errno: ::c_int,
        pub si_pad: ::c_int,
        pub si_addr: *mut ::c_void,
        __pad: [u8; 232],
    }

    pub struct sigaction {
        pub sa_flags: ::c_int,
        pub sa_sigaction: ::sighandler_t,
        pub sa_mask: sigset_t,
    }

    pub struct stack_t {
        pub ss_sp: *mut ::c_void,
        pub ss_size: ::size_t,
        pub ss_flags: ::c_int,
    }

    pub struct statvfs {
        pub f_bsize: ::c_ulong,
        pub f_frsize: ::c_ulong,
        pub f_blocks: ::fsblkcnt_t,
        pub f_bfree: ::fsblkcnt_t,
        pub f_bavail: ::fsblkcnt_t,
        pub f_files: ::fsfilcnt_t,
        pub f_ffree: ::fsfilcnt_t,
        pub f_favail: ::fsfilcnt_t,
        pub f_fsid: ::c_ulong,
        pub f_basetype: [::c_char; 16],
        pub f_flag: ::c_ulong,
        pub f_namemax: ::c_ulong,
        pub f_fstr: [::c_char; 32]
    }

    pub struct sched_param {
        pub sched_priority: ::c_int,
        sched_pad: [::c_int; 8]
    }

    pub struct Dl_info {
        pub dli_fname: *const ::c_char,
        pub dli_fbase: *mut ::c_void,
        pub dli_sname: *const ::c_char,
        pub dli_saddr: *mut ::c_void,
    }

    pub struct stat {
        pub st_dev: ::dev_t,
        pub st_ino: ::ino_t,
        pub st_mode: ::mode_t,
        pub st_nlink: ::nlink_t,
        pub st_uid: ::uid_t,
        pub st_gid: ::gid_t,
        pub st_rdev: ::dev_t,
        pub st_size: ::off_t,
        pub st_atime: ::time_t,
        pub st_atime_nsec: ::c_long,
        pub st_mtime: ::time_t,
        pub st_mtime_nsec: ::c_long,
        pub st_ctime: ::time_t,
        pub st_ctime_nsec: ::c_long,
        pub st_blksize: ::blksize_t,
        pub st_blocks: ::blkcnt_t,
        __unused: [::c_char; 16]
    }

    pub struct termios {
        pub c_iflag: ::tcflag_t,
        pub c_oflag: ::tcflag_t,
        pub c_cflag: ::tcflag_t,
        pub c_lflag: ::tcflag_t,
        pub c_cc: [::cc_t; ::NCCS]
    }

    pub struct lconv {
        pub decimal_point: *mut ::c_char,
        pub thousands_sep: *mut ::c_char,
        pub grouping: *mut ::c_char,
        pub int_curr_symbol: *mut ::c_char,
        pub currency_symbol: *mut ::c_char,
        pub mon_decimal_point: *mut ::c_char,
        pub mon_thousands_sep: *mut ::c_char,
        pub mon_grouping: *mut ::c_char,
        pub positive_sign: *mut ::c_char,
        pub negative_sign: *mut ::c_char,
        pub int_frac_digits: ::c_char,
        pub frac_digits: ::c_char,
        pub p_cs_precedes: ::c_char,
        pub p_sep_by_space: ::c_char,
        pub n_cs_precedes: ::c_char,
        pub n_sep_by_space: ::c_char,
        pub p_sign_posn: ::c_char,
        pub n_sign_posn: ::c_char,
        pub int_p_cs_precedes: ::c_char,
        pub int_p_sep_by_space: ::c_char,
        pub int_n_cs_precedes: ::c_char,
        pub int_n_sep_by_space: ::c_char,
        pub int_p_sign_posn: ::c_char,
        pub int_n_sign_posn: ::c_char,
    }

    pub struct sem_t {
        pub sem_count: u32,
        pub sem_type: u16,
        pub sem_magic: u16,
        pub sem_pad1: [u64; 3],
        pub sem_pad2: [u64; 2]
    }

    pub struct flock {
        pub l_type: ::c_short,
        pub l_whence: ::c_short,
        pub l_start: ::off_t,
        pub l_len: ::off_t,
        pub l_sysid: ::c_int,
        pub l_pid: ::pid_t,
        pub l_pad: [::c_long; 4]
    }

    pub struct if_nameindex {
        pub if_index: ::c_uint,
        pub if_name: *mut ::c_char,
    }
}

pub const LC_CTYPE: ::c_int = 0;
pub const LC_NUMERIC: ::c_int = 1;
pub const LC_TIME: ::c_int = 2;
pub const LC_COLLATE: ::c_int = 3;
pub const LC_MONETARY: ::c_int = 4;
pub const LC_MESSAGES: ::c_int = 5;
pub const LC_ALL: ::c_int = 6;
pub const LC_CTYPE_MASK: ::c_int = (1 << LC_CTYPE);
pub const LC_NUMERIC_MASK: ::c_int = (1 << LC_NUMERIC);
pub const LC_TIME_MASK: ::c_int = (1 << LC_TIME);
pub const LC_COLLATE_MASK: ::c_int = (1 << LC_COLLATE);
pub const LC_MONETARY_MASK: ::c_int = (1 << LC_MONETARY);
pub const LC_MESSAGES_MASK: ::c_int = (1 << LC_MESSAGES);
pub const LC_ALL_MASK: ::c_int = LC_CTYPE_MASK
                               | LC_NUMERIC_MASK
                               | LC_TIME_MASK
                               | LC_COLLATE_MASK
                               | LC_MONETARY_MASK
                               | LC_MESSAGES_MASK;

pub const DAY_1: ::nl_item = 1;
pub const DAY_2: ::nl_item = 2;
pub const DAY_3: ::nl_item = 3;
pub const DAY_4: ::nl_item = 4;
pub const DAY_5: ::nl_item = 5;
pub const DAY_6: ::nl_item = 6;
pub const DAY_7: ::nl_item = 7;

pub const ABDAY_1: ::nl_item = 8;
pub const ABDAY_2: ::nl_item = 9;
pub const ABDAY_3: ::nl_item = 10;
pub const ABDAY_4: ::nl_item = 11;
pub const ABDAY_5: ::nl_item = 12;
pub const ABDAY_6: ::nl_item = 13;
pub const ABDAY_7: ::nl_item = 14;

pub const MON_1: ::nl_item = 15;
pub const MON_2: ::nl_item = 16;
pub const MON_3: ::nl_item = 17;
pub const MON_4: ::nl_item = 18;
pub const MON_5: ::nl_item = 19;
pub const MON_6: ::nl_item = 20;
pub const MON_7: ::nl_item = 21;
pub const MON_8: ::nl_item = 22;
pub const MON_9: ::nl_item = 23;
pub const MON_10: ::nl_item = 24;
pub const MON_11: ::nl_item = 25;
pub const MON_12: ::nl_item = 26;

pub const ABMON_1: ::nl_item = 27;
pub const ABMON_2: ::nl_item = 28;
pub const ABMON_3: ::nl_item = 29;
pub const ABMON_4: ::nl_item = 30;
pub const ABMON_5: ::nl_item = 31;
pub const ABMON_6: ::nl_item = 32;
pub const ABMON_7: ::nl_item = 33;
pub const ABMON_8: ::nl_item = 34;
pub const ABMON_9: ::nl_item = 35;
pub const ABMON_10: ::nl_item = 36;
pub const ABMON_11: ::nl_item = 37;
pub const ABMON_12: ::nl_item = 38;

pub const RADIXCHAR: ::nl_item = 39;
pub const THOUSEP: ::nl_item = 40;
pub const YESSTR: ::nl_item = 41;
pub const NOSTR: ::nl_item = 42;
pub const CRNCYSTR: ::nl_item = 43;

pub const D_T_FMT: ::nl_item = 44;
pub const D_FMT: ::nl_item = 45;
pub const T_FMT: ::nl_item = 46;
pub const AM_STR: ::nl_item = 47;
pub const PM_STR: ::nl_item = 48;

pub const CODESET: ::nl_item = 49;
pub const T_FMT_AMPM: ::nl_item = 50;
pub const ERA: ::nl_item = 51;
pub const ERA_D_FMT: ::nl_item = 52;
pub const ERA_D_T_FMT: ::nl_item = 53;
pub const ERA_T_FMT: ::nl_item = 54;
pub const ALT_DIGITS: ::nl_item = 55;
pub const YESEXPR: ::nl_item = 56;
pub const NOEXPR: ::nl_item = 57;
pub const _DATE_FMT: ::nl_item = 58;
pub const MAXSTRMSG: ::nl_item = 58;

pub const PATH_MAX: ::c_int = 1024;

pub const SA_ONSTACK: ::c_int = 0x00000001;
pub const SA_RESETHAND: ::c_int = 0x00000002;
pub const SA_RESTART: ::c_int = 0x00000004;
pub const SA_SIGINFO: ::c_int = 0x00000008;
pub const SA_NODEFER: ::c_int = 0x00000010;
pub const SA_NOCLDWAIT: ::c_int = 0x00010000;
pub const SA_NOCLDSTOP: ::c_int = 0x00020000;

pub const SS_ONSTACK: ::c_int = 1;
pub const SS_DISABLE: ::c_int = 2;

pub const FIONBIO: ::c_int = 0x8004667e;

pub const SIGCHLD: ::c_int = 18;
pub const SIGBUS: ::c_int = 10;
pub const SIGINFO: ::c_int = 41;
pub const SIG_BLOCK: ::c_int = 1;
pub const SIG_UNBLOCK: ::c_int = 2;
pub const SIG_SETMASK: ::c_int = 3;

pub const IPV6_MULTICAST_LOOP: ::c_int = 0x8;
pub const IPV6_V6ONLY: ::c_int = 0x27;

pub const FD_SETSIZE: usize = 1024;

pub const ST_RDONLY: ::c_ulong = 1;
pub const ST_NOSUID: ::c_ulong = 2;

pub const NI_MAXHOST: ::socklen_t = 1025;

pub const EXIT_FAILURE: ::c_int = 1;
pub const EXIT_SUCCESS: ::c_int = 0;
pub const RAND_MAX: ::c_int = 32767;
pub const EOF: ::c_int = -1;
pub const SEEK_SET: ::c_int = 0;
pub const SEEK_CUR: ::c_int = 1;
pub const SEEK_END: ::c_int = 2;
pub const _IOFBF: ::c_int = 0;
pub const _IONBF: ::c_int = 4;
pub const _IOLBF: ::c_int = 64;
pub const BUFSIZ: ::c_uint = 1024;
pub const FOPEN_MAX: ::c_uint = 20;
pub const FILENAME_MAX: ::c_uint = 1024;
pub const L_tmpnam: ::c_uint = 25;
pub const TMP_MAX: ::c_uint = 17576;

pub const O_RDONLY: ::c_int = 0;
pub const O_WRONLY: ::c_int = 1;
pub const O_RDWR: ::c_int = 2;
pub const O_APPEND: ::c_int = 8;
pub const O_CREAT: ::c_int = 256;
pub const O_EXCL: ::c_int = 1024;
pub const O_NOCTTY: ::c_int = 2048;
pub const O_TRUNC: ::c_int = 512;
pub const O_CLOEXEC: ::c_int = 0x800000;
pub const O_ACCMODE: ::c_int = 0x600003;
pub const S_IFIFO: mode_t = 4096;
pub const S_IFCHR: mode_t = 8192;
pub const S_IFBLK: mode_t = 24576;
pub const S_IFDIR: mode_t = 16384;
pub const S_IFREG: mode_t = 32768;
pub const S_IFLNK: mode_t = 40960;
pub const S_IFSOCK: mode_t = 49152;
pub const S_IFMT: mode_t = 61440;
pub const S_IEXEC: mode_t = 64;
pub const S_IWRITE: mode_t = 128;
pub const S_IREAD: mode_t = 256;
pub const S_IRWXU: mode_t = 448;
pub const S_IXUSR: mode_t = 64;
pub const S_IWUSR: mode_t = 128;
pub const S_IRUSR: mode_t = 256;
pub const S_IRWXG: mode_t = 56;
pub const S_IXGRP: mode_t = 8;
pub const S_IWGRP: mode_t = 16;
pub const S_IRGRP: mode_t = 32;
pub const S_IRWXO: mode_t = 7;
pub const S_IXOTH: mode_t = 1;
pub const S_IWOTH: mode_t = 2;
pub const S_IROTH: mode_t = 4;
pub const F_OK: ::c_int = 0;
pub const R_OK: ::c_int = 4;
pub const W_OK: ::c_int = 2;
pub const X_OK: ::c_int = 1;
pub const STDIN_FILENO: ::c_int = 0;
pub const STDOUT_FILENO: ::c_int = 1;
pub const STDERR_FILENO: ::c_int = 2;
pub const F_LOCK: ::c_int = 1;
pub const F_TEST: ::c_int = 3;
pub const F_TLOCK: ::c_int = 2;
pub const F_ULOCK: ::c_int = 0;
pub const F_DUPFD_CLOEXEC: ::c_int = 37;
pub const F_SETLK: ::c_int = 6;
pub const F_SETLKW: ::c_int = 7;
pub const F_GETLK: ::c_int = 14;
pub const SIGHUP: ::c_int = 1;
pub const SIGINT: ::c_int = 2;
pub const SIGQUIT: ::c_int = 3;
pub const SIGILL: ::c_int = 4;
pub const SIGABRT: ::c_int = 6;
pub const SIGEMT: ::c_int = 7;
pub const SIGFPE: ::c_int = 8;
pub const SIGKILL: ::c_int = 9;
pub const SIGSEGV: ::c_int = 11;
pub const SIGSYS: ::c_int = 12;
pub const SIGPIPE: ::c_int = 13;
pub const SIGALRM: ::c_int = 14;
pub const SIGTERM: ::c_int = 15;
pub const SIGUSR1: ::c_int = 16;
pub const SIGUSR2: ::c_int = 17;
pub const SIGPWR: ::c_int = 19;
pub const SIGWINCH: ::c_int = 20;
pub const SIGURG: ::c_int = 21;
pub const SIGPOLL: ::c_int = 22;
pub const SIGIO: ::c_int = SIGPOLL;
pub const SIGSTOP: ::c_int = 23;
pub const SIGTSTP: ::c_int = 24;
pub const SIGCONT: ::c_int = 25;
pub const SIGTTIN: ::c_int = 26;
pub const SIGTTOU: ::c_int = 27;
pub const SIGVTALRM: ::c_int = 28;
pub const SIGPROF: ::c_int = 29;
pub const SIGXCPU: ::c_int = 30;
pub const SIGXFSZ: ::c_int = 31;

pub const WNOHANG: ::c_int = 0x40;
pub const WUNTRACED: ::c_int = 0x04;

pub const PROT_NONE: ::c_int = 0;
pub const PROT_READ: ::c_int = 1;
pub const PROT_WRITE: ::c_int = 2;
pub const PROT_EXEC: ::c_int = 4;

pub const MAP_SHARED: ::c_int = 0x0001;
pub const MAP_PRIVATE: ::c_int = 0x0002;
pub const MAP_FIXED: ::c_int = 0x0010;
pub const MAP_NORESERVE: ::c_int = 0x40;
pub const MAP_ANON: ::c_int = 0x0100;
pub const MAP_RENAME: ::c_int = 0x20;
pub const MAP_ALIGN: ::c_int = 0x200;
pub const MAP_TEXT: ::c_int = 0x400;
pub const MAP_INITDATA: ::c_int = 0x800;
pub const MAP_FAILED: *mut ::c_void = !0 as *mut ::c_void;

pub const MCL_CURRENT: ::c_int = 0x0001;
pub const MCL_FUTURE: ::c_int = 0x0002;

pub const MS_SYNC: ::c_int = 0x0004;
pub const MS_ASYNC: ::c_int = 0x0001;
pub const MS_INVALIDATE: ::c_int = 0x0002;
pub const MS_INVALCURPROC: ::c_int = 0x0008;

pub const EPERM: ::c_int = 1;
pub const ENOENT: ::c_int = 2;
pub const ESRCH: ::c_int = 3;
pub const EINTR: ::c_int = 4;
pub const EIO: ::c_int = 5;
pub const ENXIO: ::c_int = 6;
pub const E2BIG: ::c_int = 7;
pub const ENOEXEC: ::c_int = 8;
pub const EBADF: ::c_int = 9;
pub const ECHILD: ::c_int = 10;
pub const EDEADLK: ::c_int = 45;
pub const ENOMEM: ::c_int = 12;
pub const EACCES: ::c_int = 13;
pub const EFAULT: ::c_int = 14;
pub const ENOTBLK: ::c_int = 15;
pub const EBUSY: ::c_int = 16;
pub const EEXIST: ::c_int = 17;
pub const EXDEV: ::c_int = 18;
pub const ENODEV: ::c_int = 19;
pub const ENOTDIR: ::c_int = 20;
pub const EISDIR: ::c_int = 21;
pub const EINVAL: ::c_int = 22;
pub const ENFILE: ::c_int = 23;
pub const EMFILE: ::c_int = 24;
pub const ENOTTY: ::c_int = 25;
pub const ETXTBSY: ::c_int = 26;
pub const EFBIG: ::c_int = 27;
pub const ENOSPC: ::c_int = 28;
pub const ESPIPE: ::c_int = 29;
pub const EROFS: ::c_int = 30;
pub const EMLINK: ::c_int = 31;
pub const EPIPE: ::c_int = 32;
pub const EDOM: ::c_int = 33;
pub const ERANGE: ::c_int = 34;
pub const EAGAIN: ::c_int = 11;
pub const EWOULDBLOCK: ::c_int = 11;
pub const EINPROGRESS: ::c_int = 150;
pub const EALREADY: ::c_int = 149;
pub const ENOTSOCK: ::c_int = 95;
pub const EDESTADDRREQ: ::c_int = 96;
pub const EMSGSIZE: ::c_int = 97;
pub const EPROTOTYPE: ::c_int = 98;
pub const ENOPROTOOPT: ::c_int = 99;
pub const EPROTONOSUPPORT: ::c_int = 120;
pub const ESOCKTNOSUPPORT: ::c_int = 121;
pub const EOPNOTSUPP: ::c_int = 122;
pub const EPFNOSUPPORT: ::c_int = 123;
pub const EAFNOSUPPORT: ::c_int = 124;
pub const EADDRINUSE: ::c_int = 125;
pub const EADDRNOTAVAIL: ::c_int = 126;
pub const ENETDOWN: ::c_int = 127;
pub const ENETUNREACH: ::c_int = 128;
pub const ENETRESET: ::c_int = 129;
pub const ECONNABORTED: ::c_int = 130;
pub const ECONNRESET: ::c_int = 131;
pub const ENOBUFS: ::c_int = 132;
pub const EISCONN: ::c_int = 133;
pub const ENOTCONN: ::c_int = 134;
pub const ESHUTDOWN: ::c_int = 143;
pub const ETOOMANYREFS: ::c_int = 144;
pub const ETIMEDOUT: ::c_int = 145;
pub const ECONNREFUSED: ::c_int = 146;
pub const ELOOP: ::c_int = 90;
pub const ENAMETOOLONG: ::c_int = 78;
pub const EHOSTDOWN: ::c_int = 147;
pub const EHOSTUNREACH: ::c_int = 148;
pub const ENOTEMPTY: ::c_int = 93;
pub const EUSERS: ::c_int = 94;
pub const EDQUOT: ::c_int = 49;
pub const ESTALE: ::c_int = 151;
pub const EREMOTE: ::c_int = 66;
pub const ENOLCK: ::c_int = 46;
pub const ENOSYS: ::c_int = 89;
pub const EIDRM: ::c_int = 36;
pub const ENOMSG: ::c_int = 35;
pub const EOVERFLOW: ::c_int = 79;
pub const ECANCELED: ::c_int = 47;
pub const EILSEQ: ::c_int = 88;
pub const EBADMSG: ::c_int = 77;
pub const EMULTIHOP: ::c_int = 74;
pub const ENOLINK: ::c_int = 67;
pub const EPROTO: ::c_int = 71;

pub const EAI_SYSTEM: ::c_int = 11;

pub const F_DUPFD: ::c_int = 0;
pub const F_GETFD: ::c_int = 1;
pub const F_SETFD: ::c_int = 2;
pub const F_GETFL: ::c_int = 3;
pub const F_SETFL: ::c_int = 4;

pub const SIGTRAP: ::c_int = 5;

pub const GLOB_APPEND  : ::c_int = 32;
pub const GLOB_DOOFFS  : ::c_int = 16;
pub const GLOB_ERR     : ::c_int = 1;
pub const GLOB_MARK    : ::c_int = 2;
pub const GLOB_NOCHECK : ::c_int = 8;
pub const GLOB_NOSORT  : ::c_int = 4;
pub const GLOB_NOESCAPE: ::c_int = 64;

pub const GLOB_NOSPACE : ::c_int = -2;
pub const GLOB_ABORTED : ::c_int = -1;
pub const GLOB_NOMATCH : ::c_int = -3;

pub const POSIX_MADV_NORMAL: ::c_int = 0;
pub const POSIX_MADV_RANDOM: ::c_int = 1;
pub const POSIX_MADV_SEQUENTIAL: ::c_int = 2;
pub const POSIX_MADV_WILLNEED: ::c_int = 3;
pub const POSIX_MADV_DONTNEED: ::c_int = 4;

pub const _SC_IOV_MAX: ::c_int = 77;
pub const _SC_GETGR_R_SIZE_MAX: ::c_int = 569;
pub const _SC_GETPW_R_SIZE_MAX: ::c_int = 570;
pub const _SC_LOGIN_NAME_MAX: ::c_int = 571;
pub const _SC_MQ_PRIO_MAX: ::c_int = 30;
pub const _SC_THREAD_ATTR_STACKADDR: ::c_int = 577;
pub const _SC_THREAD_ATTR_STACKSIZE: ::c_int = 578;
pub const _SC_THREAD_DESTRUCTOR_ITERATIONS: ::c_int = 568;
pub const _SC_THREAD_KEYS_MAX: ::c_int = 572;
pub const _SC_THREAD_PRIO_INHERIT: ::c_int = 580;
pub const _SC_THREAD_PRIO_PROTECT: ::c_int = 581;
pub const _SC_THREAD_PRIORITY_SCHEDULING: ::c_int = 579;
pub const _SC_THREAD_PROCESS_SHARED: ::c_int = 582;
pub const _SC_THREAD_SAFE_FUNCTIONS: ::c_int = 583;
pub const _SC_THREAD_STACK_MIN: ::c_int = 573;
pub const _SC_THREAD_THREADS_MAX: ::c_int = 574;
pub const _SC_THREADS: ::c_int = 576;
pub const _SC_TTY_NAME_MAX: ::c_int = 575;
pub const _SC_ATEXIT_MAX: ::c_int = 76;
pub const _SC_XOPEN_CRYPT: ::c_int = 62;
pub const _SC_XOPEN_ENH_I18N: ::c_int = 63;
pub const _SC_XOPEN_LEGACY: ::c_int = 717;
pub const _SC_XOPEN_REALTIME: ::c_int = 718;
pub const _SC_XOPEN_REALTIME_THREADS: ::c_int = 719;
pub const _SC_XOPEN_SHM: ::c_int = 64;
pub const _SC_XOPEN_UNIX: ::c_int = 78;
pub const _SC_XOPEN_VERSION: ::c_int = 12;
pub const _SC_XOPEN_XCU_VERSION: ::c_int = 67;

pub const PTHREAD_CREATE_JOINABLE: ::c_int = 0;
pub const PTHREAD_CREATE_DETACHED: ::c_int = 0x40;
pub const PTHREAD_PROCESS_SHARED: ::c_int = 1;
pub const PTHREAD_PROCESS_PRIVATE: u16 = 0;
pub const PTHREAD_STACK_MIN: ::size_t = 4096;

pub const SIGSTKSZ: ::size_t = 8192;

// https://illumos.org/man/3c/clock_gettime
// https://github.com/illumos/illumos-gate/
//   blob/HEAD/usr/src/lib/libc/amd64/sys/__clock_gettime.s
// clock_gettime(3c) doesn't seem to accept anything other than CLOCK_REALTIME
// or __CLOCK_REALTIME0
//
// https://github.com/illumos/illumos-gate/
//   blob/HEAD/usr/src/uts/common/sys/time_impl.h
// Confusing! CLOCK_HIGHRES==CLOCK_MONOTONIC==4
// __CLOCK_REALTIME0==0 is an obsoleted version of CLOCK_REALTIME==3
pub const CLOCK_REALTIME: clockid_t = 3;
pub const CLOCK_MONOTONIC: clockid_t = 4;
pub const TIMER_RELTIME: ::c_int = 0;
pub const TIMER_ABSTIME: ::c_int = 1;

pub const RLIMIT_CPU: ::c_int = 0;
pub const RLIMIT_FSIZE: ::c_int = 1;
pub const RLIMIT_DATA: ::c_int = 2;
pub const RLIMIT_STACK: ::c_int = 3;
pub const RLIMIT_CORE: ::c_int = 4;
pub const RLIMIT_NOFILE: ::c_int = 5;
pub const RLIMIT_VMEM: ::c_int = 6;
pub const RLIMIT_AS: ::c_int = RLIMIT_VMEM;

pub const RLIM_NLIMITS: rlim_t = 7;
pub const RLIM_INFINITY: rlim_t = 0x7fffffff;

pub const RUSAGE_SELF: ::c_int = 0;
pub const RUSAGE_CHILDREN: ::c_int = -1;

pub const MADV_NORMAL: ::c_int = 0;
pub const MADV_RANDOM: ::c_int = 1;
pub const MADV_SEQUENTIAL: ::c_int = 2;
pub const MADV_WILLNEED: ::c_int = 3;
pub const MADV_DONTNEED: ::c_int = 4;
pub const MADV_FREE: ::c_int = 5;

pub const AF_INET: ::c_int = 2;
pub const AF_INET6: ::c_int = 26;
pub const AF_UNIX: ::c_int = 1;
pub const SOCK_DGRAM: ::c_int = 1;
pub const SOCK_STREAM: ::c_int = 2;
pub const SOCK_RAW: ::c_int = 4;
pub const SOCK_RDM: ::c_int = 5;
pub const SOCK_SEQPACKET: ::c_int = 6;
pub const IPPROTO_TCP: ::c_int = 6;
pub const IPPROTO_IP: ::c_int = 0;
pub const IPPROTO_IPV6: ::c_int = 41;
pub const IP_MULTICAST_TTL: ::c_int = 17;
pub const IP_MULTICAST_LOOP: ::c_int = 18;
pub const IP_TTL: ::c_int = 4;
pub const IP_HDRINCL: ::c_int = 2;
pub const IP_ADD_MEMBERSHIP: ::c_int = 19;
pub const IP_DROP_MEMBERSHIP: ::c_int = 20;
pub const IPV6_JOIN_GROUP: ::c_int = 9;
pub const IPV6_LEAVE_GROUP: ::c_int = 10;

pub const TCP_NODELAY: ::c_int = 1;
pub const TCP_KEEPIDLE: ::c_int = 34;
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
pub const SO_SNDBUF: ::c_int = 0x1001;
pub const SO_RCVBUF: ::c_int = 0x1002;
pub const SO_SNDLOWAT: ::c_int = 0x1003;
pub const SO_RCVLOWAT: ::c_int = 0x1004;
pub const SO_SNDTIMEO: ::c_int = 0x1005;
pub const SO_RCVTIMEO: ::c_int = 0x1006;
pub const SO_ERROR: ::c_int = 0x1007;
pub const SO_TYPE: ::c_int = 0x1008;

pub const IFF_LOOPBACK: ::c_int = 0x8;

pub const SHUT_RD: ::c_int = 0;
pub const SHUT_WR: ::c_int = 1;
pub const SHUT_RDWR: ::c_int = 2;

pub const LOCK_SH: ::c_int = 1;
pub const LOCK_EX: ::c_int = 2;
pub const LOCK_NB: ::c_int = 4;
pub const LOCK_UN: ::c_int = 8;

pub const O_SYNC: ::c_int = 16;
pub const O_NONBLOCK: ::c_int = 128;

pub const IPPROTO_RAW: ::c_int = 255;

pub const _SC_ARG_MAX: ::c_int = 1;
pub const _SC_CHILD_MAX: ::c_int = 2;
pub const _SC_CLK_TCK: ::c_int = 3;
pub const _SC_NGROUPS_MAX: ::c_int = 4;
pub const _SC_OPEN_MAX: ::c_int = 5;
pub const _SC_JOB_CONTROL: ::c_int = 6;
pub const _SC_SAVED_IDS: ::c_int = 7;
pub const _SC_VERSION: ::c_int = 8;
pub const _SC_PAGESIZE: ::c_int = 11;
pub const _SC_PAGE_SIZE: ::c_int = _SC_PAGESIZE;
pub const _SC_NPROCESSORS_ONLN: ::c_int = 15;
pub const _SC_STREAM_MAX: ::c_int = 16;
pub const _SC_TZNAME_MAX: ::c_int = 17;
pub const _SC_AIO_LISTIO_MAX: ::c_int = 18;
pub const _SC_AIO_MAX: ::c_int = 19;
pub const _SC_BC_BASE_MAX: ::c_int = 54;
pub const _SC_BC_DIM_MAX: ::c_int = 55;
pub const _SC_BC_SCALE_MAX: ::c_int = 56;
pub const _SC_BC_STRING_MAX: ::c_int = 57;
pub const _SC_COLL_WEIGHTS_MAX: ::c_int = 58;
pub const _SC_EXPR_NEST_MAX: ::c_int = 59;
pub const _SC_LINE_MAX: ::c_int = 60;
pub const _SC_RE_DUP_MAX: ::c_int = 61;
pub const _SC_2_VERSION: ::c_int = 53;
pub const _SC_2_C_BIND: ::c_int = 45;
pub const _SC_2_C_DEV: ::c_int = 46;
pub const _SC_2_CHAR_TERM: ::c_int = 66;
pub const _SC_2_FORT_DEV: ::c_int = 48;
pub const _SC_2_FORT_RUN: ::c_int = 49;
pub const _SC_2_LOCALEDEF: ::c_int = 50;
pub const _SC_2_SW_DEV: ::c_int = 51;
pub const _SC_2_UPE: ::c_int = 52;
pub const _SC_ASYNCHRONOUS_IO: ::c_int = 21;
pub const _SC_MAPPED_FILES: ::c_int = 24;
pub const _SC_MEMLOCK: ::c_int = 25;
pub const _SC_MEMLOCK_RANGE: ::c_int = 26;
pub const _SC_MEMORY_PROTECTION: ::c_int = 27;
pub const _SC_MESSAGE_PASSING: ::c_int = 28;
pub const _SC_PRIORITIZED_IO: ::c_int = 31;
pub const _SC_PRIORITY_SCHEDULING: ::c_int = 32;
pub const _SC_REALTIME_SIGNALS: ::c_int = 33;
pub const _SC_SEMAPHORES: ::c_int = 35;
pub const _SC_FSYNC: ::c_int = 23;
pub const _SC_SHARED_MEMORY_OBJECTS: ::c_int = 38;
pub const _SC_SYNCHRONIZED_IO: ::c_int = 42;
pub const _SC_TIMERS: ::c_int = 43;
pub const _SC_AIO_PRIO_DELTA_MAX: ::c_int = 20;
pub const _SC_DELAYTIMER_MAX: ::c_int = 22;
pub const _SC_MQ_OPEN_MAX: ::c_int = 29;
pub const _SC_RTSIG_MAX: ::c_int = 34;
pub const _SC_SEM_NSEMS_MAX: ::c_int = 36;
pub const _SC_SEM_VALUE_MAX: ::c_int = 37;
pub const _SC_SIGQUEUE_MAX: ::c_int = 39;
pub const _SC_TIMER_MAX: ::c_int = 44;

pub const _MUTEX_MAGIC: u16 = 0x4d58; // MX
pub const _COND_MAGIC: u16 = 0x4356;  // CV
pub const _RWL_MAGIC: u16 = 0x5257;   // RW

pub const NCCS: usize = 19;

pub const LOG_CRON: ::c_int = 15 << 3;

pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
    __pthread_mutex_flag1: 0,
    __pthread_mutex_flag2: 0,
    __pthread_mutex_ceiling: 0,
    __pthread_mutex_type: PTHREAD_PROCESS_PRIVATE,
    __pthread_mutex_magic: _MUTEX_MAGIC,
    __pthread_mutex_lock: 0,
    __pthread_mutex_data: 0
};
pub const PTHREAD_COND_INITIALIZER: pthread_cond_t = pthread_cond_t {
    __pthread_cond_flag: [0; 4],
    __pthread_cond_type: PTHREAD_PROCESS_PRIVATE,
    __pthread_cond_magic: _COND_MAGIC,
    __pthread_cond_data: 0
};
pub const PTHREAD_RWLOCK_INITIALIZER: pthread_rwlock_t = pthread_rwlock_t {
    __pthread_rwlock_readers: 0,
    __pthread_rwlock_type: PTHREAD_PROCESS_PRIVATE,
    __pthread_rwlock_magic: _RWL_MAGIC,
    __pthread_rwlock_mutex: PTHREAD_MUTEX_INITIALIZER,
    __pthread_rwlock_readercv: PTHREAD_COND_INITIALIZER,
    __pthread_rwlock_writercv: PTHREAD_COND_INITIALIZER
};
pub const PTHREAD_MUTEX_NORMAL: ::c_int = 0;
pub const PTHREAD_MUTEX_ERRORCHECK: ::c_int = 2;
pub const PTHREAD_MUTEX_RECURSIVE: ::c_int = 4;
pub const PTHREAD_MUTEX_DEFAULT: ::c_int = PTHREAD_MUTEX_NORMAL;

pub const RTLD_NEXT: *mut ::c_void = -1isize as *mut ::c_void;
pub const RTLD_DEFAULT: *mut ::c_void = -2isize as *mut ::c_void;
pub const RTLD_SELF: *mut ::c_void = -3isize as *mut ::c_void;
pub const RTLD_PROBE: *mut ::c_void = -4isize as *mut ::c_void;

pub const RTLD_NOW: ::c_int = 0x2;
pub const RTLD_NOLOAD: ::c_int = 0x4;
pub const RTLD_GLOBAL: ::c_int = 0x100;
pub const RTLD_LOCAL: ::c_int = 0x0;
pub const RTLD_PARENT: ::c_int = 0x200;
pub const RTLD_GROUP: ::c_int = 0x400;
pub const RTLD_WORLD: ::c_int = 0x800;
pub const RTLD_NODELETE: ::c_int = 0x1000;
pub const RTLD_FIRST: ::c_int = 0x2000;
pub const RTLD_CONFGEN: ::c_int = 0x10000;

f! {
    pub fn FD_CLR(fd: ::c_int, set: *mut fd_set) -> () {
        let fd = fd as usize;
        (*set).fds_bits[fd / 32] &= !(1 << (fd % 32));
        return
    }

    pub fn FD_ISSET(fd: ::c_int, set: *mut fd_set) -> bool {
        let fd = fd as usize;
        return ((*set).fds_bits[fd / 32] & (1 << (fd % 32))) != 0
    }

    pub fn FD_SET(fd: ::c_int, set: *mut fd_set) -> () {
        let fd = fd as usize;
        (*set).fds_bits[fd / 32] |= 1 << (fd % 32);
        return
    }

    pub fn FD_ZERO(set: *mut fd_set) -> () {
        for slot in (*set).fds_bits.iter_mut() {
            *slot = 0;
        }
    }

    pub fn WIFEXITED(status: ::c_int) -> bool {
        (status & 0xFF) == 0
    }

    pub fn WEXITSTATUS(status: ::c_int) -> ::c_int {
        (status >> 8) & 0xFF
    }

    pub fn WTERMSIG(status: ::c_int) -> ::c_int {
        status & 0x7F
    }
}

extern {
    pub fn getifaddrs(ifap: *mut *mut ::ifaddrs) -> ::c_int;
    pub fn freeifaddrs(ifa: *mut ::ifaddrs);

    pub fn stack_getbounds(sp: *mut ::stack_t) -> ::c_int;
    pub fn mincore(addr: *const ::c_void, len: ::size_t,
                   vec: *mut c_char) -> ::c_int;
    pub fn setgroups(ngroups: ::c_int,
                     ptr: *const ::gid_t) -> ::c_int;
    pub fn ioctl(fildes: ::c_int, request: ::c_int, ...) -> ::c_int;
    pub fn mprotect(addr: *const ::c_void, len: ::size_t, prot: ::c_int)
                    -> ::c_int;
    pub fn clock_getres(clk_id: clockid_t, tp: *mut ::timespec) -> ::c_int;
    pub fn clock_gettime(clk_id: clockid_t, tp: *mut ::timespec) -> ::c_int;
    pub fn clock_nanosleep(clk_id: clockid_t,
                           flags: ::c_int,
                           rqtp: *const ::timespec,
                           rmtp:  *mut ::timespec) -> ::c_int;
    pub fn getnameinfo(sa: *const ::sockaddr,
                       salen: ::socklen_t,
                       host: *mut ::c_char,
                       hostlen: ::socklen_t,
                       serv: *mut ::c_char,
                       sevlen: ::socklen_t,
                       flags: ::c_int) -> ::c_int;
    pub fn getpwnam_r(name: *const ::c_char,
                      pwd: *mut passwd,
                      buf: *mut ::c_char,
                      buflen: ::c_int) -> *const passwd;
    pub fn getpwuid_r(uid: ::uid_t,
                      pwd: *mut passwd,
                      buf: *mut ::c_char,
                      buflen: ::c_int) -> *const passwd;
    pub fn setpwent();
    pub fn getpwent() -> *mut passwd;
    pub fn readdir(dirp: *mut ::DIR) -> *const ::dirent;
    pub fn fdatasync(fd: ::c_int) -> ::c_int;
    pub fn nl_langinfo_l(item: ::nl_item, locale: ::locale_t) -> *mut ::c_char;
    pub fn duplocale(base: ::locale_t) -> ::locale_t;
    pub fn freelocale(loc: ::locale_t);
    pub fn newlocale(mask: ::c_int,
                     locale: *const ::c_char,
                     base: ::locale_t) -> ::locale_t;
    pub fn uselocale(loc: ::locale_t) -> ::locale_t;
    pub fn getprogname() -> *const ::c_char;
    pub fn setprogname(name: *const ::c_char);
    pub fn getloadavg(loadavg: *mut ::c_double, nelem: ::c_int) -> ::c_int;
    pub fn getpriority(which: ::c_int, who: ::c_int) -> ::c_int;
    pub fn setpriority(which: ::c_int, who: ::c_int, prio: ::c_int) -> ::c_int;

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
    pub fn sethostname(name: *const ::c_char, len: ::size_t) -> ::c_int;
    pub fn if_nameindex() -> *mut if_nameindex;
    pub fn if_freenameindex(ptr: *mut if_nameindex);
    pub fn pthread_condattr_getclock(attr: *const pthread_condattr_t,
                                     clock_id: *mut clockid_t) -> ::c_int;
    pub fn pthread_condattr_setclock(attr: *mut pthread_condattr_t,
                                     clock_id: clockid_t) -> ::c_int;
    pub fn sem_timedwait(sem: *mut sem_t,
                         abstime: *const ::timespec) -> ::c_int;
    pub fn pthread_mutex_timedlock(lock: *mut pthread_mutex_t,
                                   abstime: *const ::timespec) -> ::c_int;
}
