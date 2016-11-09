use dox::mem;

pub type rlim_t = ::uintptr_t;
pub type sa_family_t = u8;
pub type pthread_key_t = ::c_int;
pub type nfds_t = ::c_long;
pub type tcflag_t = ::c_uint;
pub type speed_t = ::c_uint;
pub type c_char = i8;
pub type clock_t = i32;
pub type clockid_t = i32;
pub type time_t = i32;
pub type suseconds_t = i32;
pub type wchar_t = i32;
pub type off_t = i64;
pub type ino_t = i64;
pub type blkcnt_t = i64;
pub type blksize_t = i32;
pub type dev_t = i32;
pub type mode_t = u32;
pub type nlink_t = i32;
pub type useconds_t = u32;
pub type socklen_t = u32;
pub type pthread_t = ::uintptr_t;
pub type pthread_mutexattr_t = ::uintptr_t;
pub type sigset_t = u64;
pub type fsblkcnt_t = i64;
pub type fsfilcnt_t = i64;
pub type pthread_attr_t = *mut ::c_void;
pub type nl_item = ::c_int;

pub enum timezone {}

s! {
    pub struct sockaddr {
        pub sa_len: u8,
        pub sa_family: sa_family_t,
        pub sa_data: [::c_char; 30],
    }

    pub struct sockaddr_in {
        pub sin_len: u8,
        pub sin_family: sa_family_t,
        pub sin_port: ::in_port_t,
        pub sin_addr: ::in_addr,
        pub sin_zero: [u8; 24],
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
        pub sun_path: [::c_char; 126]
    }

    pub struct sockaddr_storage {
        pub ss_len: u8,
        pub ss_family: sa_family_t,
        __ss_pad1: [u8; 6],
        __ss_pad2: u64,
        __ss_pad3: [u8; 112],
    }

    pub struct addrinfo {
        pub ai_flags: ::c_int,
        pub ai_family: ::c_int,
        pub ai_socktype: ::c_int,
        pub ai_protocol: ::c_int,
        pub ai_addrlen: socklen_t,
        pub ai_canonname: *mut c_char,
        pub ai_addr: *mut ::sockaddr,
        pub ai_next: *mut addrinfo,
    }

    pub struct fd_set {
        fds_bits: [c_ulong; FD_SETSIZE / ULONG_SIZE],
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
        pub tm_zone: *const ::c_char,
    }

    pub struct utsname {
        pub sysname: [::c_char; 32],
        pub nodename: [::c_char; 32],
        pub release: [::c_char; 32],
        pub version: [::c_char; 32],
        pub machine: [::c_char; 32],
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

    pub struct msghdr {
        pub msg_name: *mut ::c_void,
        pub msg_namelen: ::socklen_t,
        pub msg_iov: *mut ::iovec,
        pub msg_iovlen: ::c_int,
        pub msg_control: *mut ::c_void,
        pub msg_controllen: ::socklen_t,
        pub msg_flags: ::c_int,
    }

    pub struct Dl_info {
        pub dli_fname: *const ::c_char,
        pub dli_fbase: *mut ::c_void,
        pub dli_sname: *const ::c_char,
        pub dli_saddr: *mut ::c_void,
    }

    pub struct termios {
        pub c_iflag: ::tcflag_t,
        pub c_oflag: ::tcflag_t,
        pub c_cflag: ::tcflag_t,
        pub c_lflag: ::tcflag_t,
        pub c_line:  ::c_char,
        pub c_ispeed: ::speed_t,
        pub c_ospeed: ::speed_t,
        pub c_cc: [::cc_t; ::NCCS],
    }

    pub struct stat {
        pub st_dev: dev_t,
        pub st_ino: ino_t,
        pub st_mode: mode_t,
        pub st_nlink: nlink_t,
        pub st_uid: ::uid_t,
        pub st_gid: ::gid_t,
        pub st_size: off_t,
        pub st_rdev: dev_t,
        pub st_blksize: blksize_t,
        pub st_atime: time_t,
        pub st_atime_nsec: c_long,
        pub st_mtime: time_t,
        pub st_mtime_nsec: c_long,
        pub st_ctime: time_t,
        pub st_ctime_nsec: c_long,
        pub st_crtime: time_t,
        pub st_crtime_nsec: c_long,
        pub st_type: u32,
        pub st_blocks: blkcnt_t,
    }

    pub struct dirent {
        pub d_dev: dev_t,
        pub d_pdev: dev_t,
        pub d_ino: ino_t,
        pub d_pino: i64,
        pub d_reclen: ::c_ushort,
        pub d_name: [::c_char; 1024], // Max length is _POSIX_PATH_MAX
    }

    pub struct glob_t {
        pub gl_pathc: ::size_t,
        __unused1: ::size_t,
        pub gl_offs: ::size_t,
        __unused2: ::size_t,
        pub gl_pathv: *mut *mut c_char,

        __unused3: *mut ::c_void,
        __unused4: *mut ::c_void,
        __unused5: *mut ::c_void,
        __unused6: *mut ::c_void,
        __unused7: *mut ::c_void,
        __unused8: *mut ::c_void,
    }

    pub struct pthread_mutex_t {
        flags: u32,
        lock: i32,
        unused: i32,
        owner: i32,
        owner_count: i32,
    }

    pub struct pthread_cond_t {
        flags: u32,
        unused: i32,
        mutex: *mut ::c_void,
        waiter_count: i32,
        lock: i32,
    }

    pub struct pthread_rwlock_t {
        flags: u32,
        owner: i32,
        lock_sem: i32,      // this is actually a union
        lock_count: i32,
        reader_count: i32,
        writer_count: i32,
        waiters: [*mut ::c_void; 2],
    }

    pub struct passwd {
        pub pw_name: *mut ::c_char,
        pub pw_passwd: *mut ::c_char,
        pub pw_uid: ::uid_t,
        pub pw_gid: ::gid_t,
        pub pw_dir: *mut ::c_char,
        pub pw_shell: *mut ::c_char,
        pub pw_gecos: *mut ::c_char,
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
        pub f_flag: ::c_ulong,
        pub f_namemax: ::c_ulong,
    }

    pub struct stack_t {
        pub ss_sp: *mut ::c_void,
        pub ss_size: ::size_t,
        pub ss_flags: ::c_int,
    }

    pub struct siginfo_t {
        pub si_signo: ::c_int,
        pub si_code: ::c_int,
        pub si_errno: ::c_int,
        pub si_pid: ::pid_t,
        pub si_uid: ::uid_t,
        pub si_addr: *mut ::c_void,
        pub si_status: ::c_int,
        pub si_band: c_long,
        pub sigval: *mut ::c_void,
    }

    pub struct sigaction {
        pub sa_sigaction: ::sighandler_t,
        pub sa_mask: ::sigset_t,
        pub sa_flags: ::c_int,
        sa_userdata: *mut ::c_void,
    }

    pub struct sem_t {
        pub se_type: i32,
        pub se_named_id: i32, // this is actually a union
        pub se_unnamed: i32,
        pub se_padding: [i32; 4],
    }

    pub struct pthread_condattr_t {
        pub process_shared: bool,
        pub clock_id: i32,
    }
}

// intentionally not public, only used for fd_set
cfg_if! {
    if #[cfg(target_pointer_width = "32")] {
        const ULONG_SIZE: usize = 32;
    } else if #[cfg(target_pointer_width = "64")] {
        const ULONG_SIZE: usize = 64;
    } else {
        // Unknown target_pointer_width
    }
}

pub const EXIT_FAILURE: ::c_int = 1;
pub const EXIT_SUCCESS: ::c_int = 0;
pub const RAND_MAX: ::c_int = 2147483647;
pub const EOF: ::c_int = -1;
pub const SEEK_SET: ::c_int = 0;
pub const SEEK_CUR: ::c_int = 1;
pub const SEEK_END: ::c_int = 2;
pub const _IOFBF: ::c_int = 0;
pub const _IONBF: ::c_int = 2;
pub const _IOLBF: ::c_int = 1;

pub const F_DUPFD: ::c_int = 0x0001;
pub const F_GETFD: ::c_int = 0x0002;
pub const F_SETFD: ::c_int = 0x0004;
pub const F_GETFL: ::c_int = 0x0008;
pub const F_SETFL: ::c_int = 0x0010;

pub const SIGTRAP: ::c_int = 22;

pub const PTHREAD_CREATE_JOINABLE: ::c_int = 0;
pub const PTHREAD_CREATE_DETACHED: ::c_int = 1;

pub const CLOCK_REALTIME: ::c_int = -1;
pub const CLOCK_MONOTONIC: ::c_int = 0;

pub const RLIMIT_CORE: ::c_int = 0;
pub const RLIMIT_CPU: ::c_int = 1;
pub const RLIMIT_DATA: ::c_int = 2;
pub const RLIMIT_FSIZE: ::c_int = 3;
pub const RLIMIT_NOFILE: ::c_int = 4;
pub const RLIMIT_AS: ::c_int = 6;
// Haiku specific
pub const RLIMIT_NOVMON: ::c_int = 7;
pub const RLIMIT_NLIMITS: ::c_int = 8;

pub const RUSAGE_SELF: ::c_int = 0;

pub const NCCS: usize = 11;

pub const O_RDONLY: ::c_int = 0x0000;
pub const O_WRONLY: ::c_int = 0x0001;
pub const O_RDWR: ::c_int = 0x0002;
pub const O_ACCMODE: ::c_int = 0x0003;

pub const O_EXCL: ::c_int = 0x0100;
pub const O_CREAT: ::c_int = 0x0200;
pub const O_TRUNC: ::c_int = 0x0400;
pub const O_NOCTTY: ::c_int = 0x1000;
pub const O_NOTRAVERSE: ::c_int = 0x2000;

pub const O_CLOEXEC: ::c_int = 0x00000040;
pub const O_NONBLOCK: ::c_int = 0x00000080;
pub const O_APPEND: ::c_int = 0x00000800;
pub const O_SYNC: ::c_int = 0x00010000;
pub const O_RSYNC: ::c_int = 0x00020000;
pub const O_DSYNC: ::c_int = 0x00040000;
pub const O_NOFOLLOW: ::c_int = 0x00080000;
pub const O_NOCACHE: ::c_int = 0x00100000;
pub const O_DIRECTORY: ::c_int = 0x00200000;

pub const S_IFIFO: ::mode_t = 61440;
pub const S_IFCHR: ::mode_t = 49152;
pub const S_IFBLK: ::mode_t = 24576;
pub const S_IFDIR: ::mode_t = 16384;
pub const S_IFREG: ::mode_t = 32768;
pub const S_IFLNK: ::mode_t = 40960;
pub const S_IFSOCK: ::mode_t = 49152;
pub const S_IFMT: ::mode_t = 61440;
pub const S_IRWXU: ::mode_t = 448;
pub const S_IXUSR: ::mode_t = 64;
pub const S_IWUSR: ::mode_t = 128;
pub const S_IRUSR: ::mode_t = 256;
pub const S_IRWXG: ::mode_t = 70;
pub const S_IXGRP: ::mode_t = 10;
pub const S_IWGRP: ::mode_t = 20;
pub const S_IRGRP: ::mode_t = 40;
pub const S_IRWXO: ::mode_t = 7;
pub const S_IXOTH: ::mode_t = 1;
pub const S_IWOTH: ::mode_t = 2;
pub const S_IROTH: ::mode_t = 4;
pub const F_OK: ::c_int = 0;
pub const R_OK: ::c_int = 4;
pub const W_OK: ::c_int = 2;
pub const X_OK: ::c_int = 1;
pub const STDIN_FILENO: ::c_int = 0;
pub const STDOUT_FILENO: ::c_int = 1;
pub const STDERR_FILENO: ::c_int = 2;
pub const SIGHUP: ::c_int = 1;
pub const SIGINT: ::c_int = 2;
pub const SIGQUIT: ::c_int = 3;
pub const SIGILL: ::c_int = 4;
pub const SIGABRT: ::c_int = 6;
pub const SIGFPE: ::c_int = 8;
pub const SIGKILL: ::c_int = 9;
pub const SIGSEGV: ::c_int = 11;
pub const SIGPIPE: ::c_int = 7;
pub const SIGALRM: ::c_int = 14;
pub const SIGTERM: ::c_int = 15;

pub const EAI_SYSTEM: ::c_int = 11;

pub const PROT_NONE: ::c_int = 0;
pub const PROT_READ: ::c_int = 1;
pub const PROT_WRITE: ::c_int = 2;
pub const PROT_EXEC: ::c_int = 4;

pub const LC_ALL: ::c_int = 0;
pub const LC_COLLATE: ::c_int = 1;
pub const LC_CTYPE: ::c_int = 2;
pub const LC_MONETARY: ::c_int = 3;
pub const LC_NUMERIC: ::c_int = 4;
pub const LC_TIME: ::c_int = 5;
pub const LC_MESSAGES: ::c_int = 6;

// TODO: Haiku does not have MAP_FILE, but libstd/os.rs requires it
pub const MAP_FILE: ::c_int = 0x00;
pub const MAP_SHARED: ::c_int = 0x01;
pub const MAP_PRIVATE: ::c_int = 0x02;
pub const MAP_FIXED: ::c_int = 0x004;

pub const MAP_FAILED: *mut ::c_void = !0 as *mut ::c_void;

pub const MS_ASYNC: ::c_int = 0x01;
pub const MS_INVALIDATE: ::c_int = 0x04;
pub const MS_SYNC: ::c_int = 0x02;

pub const EPERM : ::c_int = -2147483633;
pub const ENOENT : ::c_int = -2147459069;
pub const ESRCH : ::c_int = -2147454963;
pub const EINTR : ::c_int = -2147483638;
pub const EIO : ::c_int = -2147483647;
pub const ENXIO : ::c_int = -2147454965;
pub const E2BIG : ::c_int = -2147454975;
pub const ENOEXEC : ::c_int = -2147478782;
pub const EBADF : ::c_int = -2147459072;
pub const ECHILD : ::c_int = -2147454974;
pub const EDEADLK : ::c_int = -2147454973;
pub const ENOMEM : ::c_int = -2147454976;
pub const EACCES : ::c_int = -2147483646;
pub const EFAULT : ::c_int = -2147478783;
// pub const ENOTBLK : ::c_int = 15;
pub const EBUSY : ::c_int = -2147483634;
pub const EEXIST : ::c_int = -2147459070;
pub const EXDEV : ::c_int = -2147459061;
pub const ENODEV : ::c_int = -2147454969;
pub const ENOTDIR : ::c_int = -2147459067;
pub const EISDIR : ::c_int = -2147459063;
pub const EINVAL : ::c_int = -2147483643;
pub const ENFILE : ::c_int = -2147454970;
pub const EMFILE : ::c_int = -2147459062;
pub const ENOTTY : ::c_int = -2147454966;
pub const ETXTBSY : ::c_int = -2147454917;
pub const EFBIG : ::c_int = -2147454972;
pub const ENOSPC : ::c_int = -2147459065;
pub const ESPIPE : ::c_int = -2147454964;
pub const EROFS : ::c_int = -2147459064;
pub const EMLINK : ::c_int = -2147454971;
pub const EPIPE : ::c_int = -2147459059;
pub const EDOM : ::c_int = -2147454960;
pub const ERANGE : ::c_int = -2147454959;
pub const EAGAIN : ::c_int = -2147483637;
pub const EWOULDBLOCK : ::c_int = -2147483637;

pub const EINPROGRESS : ::c_int = -2147454940;
pub const EALREADY : ::c_int = -2147454939;
pub const ENOTSOCK : ::c_int = -2147454932;
pub const EDESTADDRREQ : ::c_int = -2147454928;
pub const EMSGSIZE : ::c_int = -2147454934;
pub const EPROTOTYPE : ::c_int = -2147454958;
pub const ENOPROTOOPT : ::c_int = -2147454942;
pub const EPROTONOSUPPORT : ::c_int = -2147454957;
pub const EOPNOTSUPP : ::c_int = -2147454933;
pub const EPFNOSUPPORT : ::c_int = -2147454956;
pub const EAFNOSUPPORT : ::c_int = -2147454955;
pub const EADDRINUSE : ::c_int = -2147454954;
pub const EADDRNOTAVAIL : ::c_int = -2147454953;
pub const ENETDOWN : ::c_int = -2147454953;
pub const ENETUNREACH : ::c_int = -2147454951;
pub const ENETRESET : ::c_int = -2147454950;
pub const ECONNABORTED : ::c_int = -2147454949;
pub const ECONNRESET : ::c_int = -2147454948;
pub const ENOBUFS : ::c_int = -2147454941;
pub const EISCONN : ::c_int = -2147454947;
pub const ENOTCONN : ::c_int = -2147454946;
pub const ESHUTDOWN : ::c_int = -2147454945;
pub const ETIMEDOUT : ::c_int = -2147483639;
pub const ECONNREFUSED : ::c_int = -2147454944;
pub const ELOOP : ::c_int = -2147459060;
pub const ENAMETOOLONG : ::c_int = -2147459068;
pub const EHOSTDOWN : ::c_int = -2147454931;
pub const EHOSTUNREACH : ::c_int = -2147454943;
pub const ENOTEMPTY : ::c_int = -2147459066;
pub const EDQUOT : ::c_int = -2147454927;
pub const ESTALE : ::c_int = -2147454936;
pub const ENOLCK : ::c_int = -2147454968;
pub const ENOSYS : ::c_int = -2147454967;
pub const EIDRM : ::c_int = -2147454926;
pub const ENOMSG : ::c_int = -2147454937;
pub const EOVERFLOW : ::c_int = -2147454935;
pub const ECANCELED : ::c_int = -2147454929;
pub const EILSEQ : ::c_int = -2147454938;
pub const ENOATTR : ::c_int = -2147454916;
pub const EBADMSG : ::c_int = -2147454930;
pub const EMULTIHOP : ::c_int = -2147454925;
pub const ENOLINK : ::c_int = -2147454923;
pub const EPROTO : ::c_int = -2147454919;

pub const IPPROTO_RAW: ::c_int = 255;

// These are prefixed with POSIX_ on Haiku
pub const MADV_NORMAL: ::c_int = 1;
pub const MADV_SEQUENTIAL: ::c_int = 2;
pub const MADV_RANDOM: ::c_int = 3;
pub const MADV_WILLNEED: ::c_int = 4;
pub const MADV_DONTNEED: ::c_int = 5;

pub const IFF_LOOPBACK: ::c_int = 0x0008;

pub const AF_UNIX: ::c_int = 9;
pub const AF_INET: ::c_int = 1;
pub const AF_INET6: ::c_int = 6;
pub const SOCK_RAW: ::c_int = 3;
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
pub const TCP_MAXSEG: ::c_int = 0x02;
pub const TCP_NOPUSH: ::c_int = 0x04;
pub const TCP_NOOPT: ::c_int = 0x08;

pub const IPV6_MULTICAST_LOOP: ::c_int = 26;
pub const IPV6_JOIN_GROUP: ::c_int = 28;
pub const IPV6_LEAVE_GROUP: ::c_int = 29;
pub const IPV6_V6ONLY: ::c_int = 30;

pub const SO_DEBUG: ::c_int = 0x00000004;

pub const MSG_NOSIGNAL: ::c_int = 0x0800;

pub const SHUT_RD: ::c_int = 0;
pub const SHUT_WR: ::c_int = 1;
pub const SHUT_RDWR: ::c_int = 2;

pub const LOCK_SH: ::c_int = 0x01;
pub const LOCK_EX: ::c_int = 0x02;
pub const LOCK_NB: ::c_int = 0x04;
pub const LOCK_UN: ::c_int = 0x08;

pub const SIGSTKSZ: ::size_t = 16384;

pub const SA_NODEFER: ::c_int = 0x08;
pub const SA_RESETHAND: ::c_int = 0x04;
pub const SA_RESTART: ::c_int = 0x10;
pub const SA_NOCLDSTOP: ::c_int = 0x01;

pub const FD_SETSIZE: usize = 1024;

pub const RTLD_NOW: ::c_int = 0x1;
pub const RTLD_DEFAULT: *mut ::c_void = 0isize as *mut ::c_void;

pub const BUFSIZ: ::c_uint = 8192;
pub const FILENAME_MAX: ::c_uint = 256;
pub const FOPEN_MAX: ::c_uint = 128;
pub const L_tmpnam: ::c_uint = 512;
pub const TMP_MAX: ::c_uint = 32768;
pub const _PC_NAME_MAX: ::c_int = 4;

pub const FIONBIO: ::c_int = 0xbe000000;

pub const _SC_IOV_MAX : ::c_int = 32;
pub const _SC_GETGR_R_SIZE_MAX : ::c_int = 25;
pub const _SC_GETPW_R_SIZE_MAX : ::c_int = 26;
pub const _SC_PAGESIZE : ::c_int = 27;
pub const _SC_THREAD_ATTR_STACKADDR : ::c_int = 48;
pub const _SC_THREAD_ATTR_STACKSIZE : ::c_int = 49;
pub const _SC_THREAD_PRIORITY_SCHEDULING : ::c_int = 50;
pub const _SC_THREAD_PROCESS_SHARED : ::c_int = 46;
pub const _SC_THREAD_STACK_MIN : ::c_int = 47;
pub const _SC_THREADS : ::c_int = 31;
pub const _SC_ATEXIT_MAX : ::c_int = 37;

pub const PTHREAD_STACK_MIN: ::size_t = 8192;

pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
    flags: 0,
    lock: 0,
    unused: -42,
    owner: -1,
    owner_count: 0,
};
pub const PTHREAD_COND_INITIALIZER: pthread_cond_t = pthread_cond_t {
    flags: 0,
    unused: -42,
    mutex: 0 as *mut _,
    waiter_count: 0,
    lock: 0,
};
pub const PTHREAD_RWLOCK_INITIALIZER: pthread_rwlock_t = pthread_rwlock_t {
    flags: 0,
    owner: 0,
    lock_sem: 0,
    lock_count: 0,
    reader_count: 0,
    writer_count: 0,
    waiters: [0 as *mut _; 2],
};

pub const PTHREAD_MUTEX_DEFAULT: ::c_int = 0;
pub const PTHREAD_MUTEX_NORMAL: ::c_int = 1;
pub const PTHREAD_MUTEX_ERRORCHECK: ::c_int = 2;
pub const PTHREAD_MUTEX_RECURSIVE: ::c_int = 3;

pub const FIOCLEX: c_ulong = 0; // TODO: does not exist on Haiku!

pub const SA_ONSTACK: c_ulong = 0x20;
pub const SA_SIGINFO: c_ulong = 0x40;
pub const SA_NOCLDWAIT: c_ulong = 0x02;

pub const SIGCHLD: ::c_int = 5;
pub const SIGBUS: ::c_int = 30;
pub const SIG_SETMASK: ::c_int = 3;

pub const RUSAGE_CHILDREN: ::c_int = -1;

pub const SOCK_STREAM: ::c_int = 1;
pub const SOCK_DGRAM: ::c_int = 2;

pub const SOL_SOCKET: ::c_int = -1;
pub const SO_ACCEPTCONN: ::c_int = 0x00000001;
pub const SO_BROADCAST: ::c_int = 0x00000002;
pub const SO_DONTROUTE: ::c_int = 0x00000008;
pub const SO_KEEPALIVE: ::c_int = 0x00000010;
pub const SO_OOBINLINE: ::c_int = 0x00000020;
pub const SO_REUSEADDR: ::c_int = 0x00000040;
pub const SO_REUSEPORT: ::c_int = 0x00000080;
pub const SO_USELOOPBACK: ::c_int = 0x00000100;
pub const SO_LINGER: ::c_int = 0x00000200;
pub const SO_SNDBUF: ::c_int = 0x40000001;
pub const SO_SNDLOWAT: ::c_int = 0x40000002;
pub const SO_SNDTIMEO: ::c_int = 0x40000003;
pub const SO_RCVBUF: ::c_int = 0x40000004;
pub const SO_RCVLOWAT: ::c_int = 0x40000005;
pub const SO_RCVTIMEO: ::c_int = 0x40000006;
pub const SO_ERROR: ::c_int = 0x40000007;
pub const SO_TYPE: ::c_int = 0x40000008;
pub const SO_NONBLOCK: ::c_int = 0x40000009;
pub const SO_BINDTODEVICE: ::c_int = 0x4000000a;
pub const SO_PEERCRED: ::c_int = 0x4000000b;

pub const NI_MAXHOST: ::size_t = 1025;

f! {
    pub fn FD_CLR(fd: ::c_int, set: *mut fd_set) -> () {
        let fd = fd as usize;
        let size = mem::size_of_val(&(*set).fds_bits[0]) * 8;
        (*set).fds_bits[fd / size] &= !(1 << (fd % size));
        return
    }

    pub fn FD_ISSET(fd: ::c_int, set: *mut fd_set) -> bool {
        let fd = fd as usize;
        let size = mem::size_of_val(&(*set).fds_bits[0]) * 8;
        return ((*set).fds_bits[fd / size] & (1 << (fd % size))) != 0
    }

    pub fn FD_SET(fd: ::c_int, set: *mut fd_set) -> () {
        let fd = fd as usize;
        let size = mem::size_of_val(&(*set).fds_bits[0]) * 8;
        (*set).fds_bits[fd / size] |= 1 << (fd % size);
        return
    }

    pub fn FD_ZERO(set: *mut fd_set) -> () {
        for slot in (*set).fds_bits.iter_mut() {
            *slot = 0;
        }
    }

    pub fn WIFEXITED(status: ::c_int) -> bool {
        (status >> 8) == 0
    }

    pub fn WEXITSTATUS(status: ::c_int) -> ::c_int {
        (status & 0xff)
    }

    pub fn WTERMSIG(status: ::c_int) -> ::c_int {
        (status >> 8) & 0xff
    }
}

extern {
    pub fn clock_gettime(clk_id: ::c_int, tp: *mut ::timespec) -> ::c_int;
    pub fn pthread_attr_getguardsize(attr: *const ::pthread_attr_t,
                                     guardsize: *mut ::size_t) -> ::c_int;
    pub fn pthread_attr_getstack(attr: *const ::pthread_attr_t,
                                 stackaddr: *mut *mut ::c_void,
                                 stacksize: *mut ::size_t) -> ::c_int;
    pub fn pthread_condattr_getclock(attr: *const pthread_condattr_t,
                                     clock_id: *mut clockid_t) -> ::c_int;
    pub fn pthread_condattr_setclock(attr: *mut pthread_condattr_t,
                                     clock_id: clockid_t) -> ::c_int;
    pub fn memalign(align: ::size_t, size: ::size_t) -> *mut ::c_void;
    pub fn setgroups(ngroups: ::size_t,
                     ptr: *const ::gid_t) -> ::c_int;
    pub fn getpwuid_r(uid: ::uid_t,
                      pwd: *mut passwd,
                      buffer: *mut ::c_char,
                      bufferSize: ::size_t,
                      result: *mut *mut passwd) -> ::c_int;
    pub fn ioctl(fd: ::c_int, request: ::c_int, ...) -> ::c_int;
    pub fn mprotect(addr: *const ::c_void, len: ::size_t, prot: ::c_int)
                    -> ::c_int;
    pub fn getnameinfo(sa: *const ::sockaddr,
                       salen: ::socklen_t,
                       host: *mut ::c_char,
                       hostlen: ::size_t,
                       serv: *mut ::c_char,
                       sevlen: ::size_t,
                       flags: ::c_int) -> ::c_int;
    pub fn pthread_mutex_timedlock(lock: *mut pthread_mutex_t,
                                   abstime: *const ::timespec) -> ::c_int;
}

cfg_if! {
    if #[cfg(target_pointer_width = "64")] {
        mod b64;
        pub use self::b64::*;
    } else {
        mod b32;
        pub use self::b32::*;
    }
}
