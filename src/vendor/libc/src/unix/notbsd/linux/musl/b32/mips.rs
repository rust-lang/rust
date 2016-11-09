pub type c_char = i8;
pub type wchar_t = ::c_int;

s! {
    pub struct stat {
        pub st_dev: ::dev_t,
        __st_padding1: [::c_long; 2],
        pub st_ino: ::ino_t,
        pub st_mode: ::mode_t,
        pub st_nlink: ::nlink_t,
        pub st_uid: ::uid_t,
        pub st_gid: ::gid_t,
        pub st_rdev: ::dev_t,
        __st_padding2: [::c_long; 2],
        pub st_size: ::off_t,
        pub st_atime: ::time_t,
        pub st_atime_nsec: ::c_long,
        pub st_mtime: ::time_t,
        pub st_mtime_nsec: ::c_long,
        pub st_ctime: ::time_t,
        pub st_ctime_nsec: ::c_long,
        pub st_blksize: ::blksize_t,
        __st_padding3: ::c_long,
        pub st_blocks: ::blkcnt_t,
        __st_padding4: [::c_long; 14],
    }

    pub struct stat64 {
        pub st_dev: ::dev_t,
        __st_padding1: [::c_long; 2],
        pub st_ino: ::ino64_t,
        pub st_mode: ::mode_t,
        pub st_nlink: ::nlink_t,
        pub st_uid: ::uid_t,
        pub st_gid: ::gid_t,
        pub st_rdev: ::dev_t,
        __st_padding2: [::c_long; 2],
        pub st_size: ::off_t,
        pub st_atime: ::time_t,
        pub st_atime_nsec: ::c_long,
        pub st_mtime: ::time_t,
        pub st_mtime_nsec: ::c_long,
        pub st_ctime: ::time_t,
        pub st_ctime_nsec: ::c_long,
        pub st_blksize: ::blksize_t,
        __st_padding3: ::c_long,
        pub st_blocks: ::blkcnt64_t,
        __st_padding4: [::c_long; 14],
    }

    pub struct stack_t {
        pub ss_sp: *mut ::c_void,
        pub ss_size: ::size_t,
        pub ss_flags: ::c_int,
    }

    pub struct shmid_ds {
        pub shm_perm: ::ipc_perm,
        pub shm_segsz: ::size_t,
        pub shm_atime: ::time_t,
        pub shm_dtime: ::time_t,
        pub shm_ctime: ::time_t,
        pub shm_cpid: ::pid_t,
        pub shm_lpid: ::pid_t,
        pub shm_nattch: ::c_ulong,
        __pad1: ::c_ulong,
        __pad2: ::c_ulong,
    }

    pub struct msqid_ds {
        pub msg_perm: ::ipc_perm,
        #[cfg(target_endian = "big")]
        __unused1: ::c_int,
        pub msg_stime: ::time_t,
        #[cfg(target_endian = "little")]
        __unused1: ::c_int,
        #[cfg(target_endian = "big")]
        __unused2: ::c_int,
        pub msg_rtime: ::time_t,
        #[cfg(target_endian = "little")]
        __unused2: ::c_int,
        #[cfg(target_endian = "big")]
        __unused3: ::c_int,
        pub msg_ctime: ::time_t,
        #[cfg(target_endian = "little")]
        __unused3: ::c_int,
        __msg_cbytes: ::c_ulong,
        pub msg_qnum: ::msgqnum_t,
        pub msg_qbytes: ::msglen_t,
        pub msg_lspid: ::pid_t,
        pub msg_lrpid: ::pid_t,
        __pad1: ::c_ulong,
        __pad2: ::c_ulong,
    }

    pub struct statfs {
        pub f_type: ::c_ulong,
        pub f_bsize: ::c_ulong,
        pub f_frsize: ::c_ulong,
        pub f_blocks: ::fsblkcnt_t,
        pub f_bfree: ::fsblkcnt_t,
        pub f_files: ::fsfilcnt_t,
        pub f_ffree: ::fsfilcnt_t,
        pub f_bavail: ::fsblkcnt_t,
        pub f_fsid: ::fsid_t,
        pub f_namelen: ::c_ulong,
        pub f_flags: ::c_ulong,
        pub f_spare: [::c_ulong; 5],
    }

    pub struct siginfo_t {
        pub si_signo: ::c_int,
        pub si_code: ::c_int,
        pub si_errno: ::c_int,
        pub _pad: [::c_int; 29],
        _align: [usize; 0],
    }
}

pub const O_DIRECT: ::c_int = 0o100000;
pub const O_DIRECTORY: ::c_int = 0o200000;
pub const O_NOFOLLOW: ::c_int = 0o400000;
pub const O_ASYNC: ::c_int = 0o10000;

pub const FIOCLEX: ::c_int = 0x6601;
pub const FIONBIO: ::c_int = 0x667E;

pub const RLIMIT_RSS: ::c_int = 7;
pub const RLIMIT_NOFILE: ::c_int = 5;
pub const RLIMIT_AS: ::c_int = 6;
pub const RLIMIT_NPROC: ::c_int = 8;
pub const RLIMIT_MEMLOCK: ::c_int = 9;

pub const O_APPEND: ::c_int = 0o010;
pub const O_CREAT: ::c_int = 0o400;
pub const O_EXCL: ::c_int = 0o2000;
pub const O_NOCTTY: ::c_int = 0o4000;
pub const O_NONBLOCK: ::c_int = 0o200;
pub const O_SYNC: ::c_int = 0o40020;
pub const O_RSYNC: ::c_int = 0o40020;
pub const O_DSYNC: ::c_int = 0o020;

pub const SOCK_NONBLOCK: ::c_int = 0o200;

pub const MAP_ANON: ::c_int = 0x800;
pub const MAP_GROWSDOWN: ::c_int = 0x1000;
pub const MAP_DENYWRITE: ::c_int = 0x2000;
pub const MAP_EXECUTABLE: ::c_int = 0x4000;
pub const MAP_LOCKED: ::c_int = 0x8000;
pub const MAP_NORESERVE: ::c_int = 0x0400;
pub const MAP_POPULATE: ::c_int = 0x10000;
pub const MAP_NONBLOCK: ::c_int = 0x20000;
pub const MAP_STACK: ::c_int = 0x40000;

pub const EDEADLK: ::c_int = 45;
pub const ENAMETOOLONG: ::c_int = 78;
pub const ENOLCK: ::c_int = 46;
pub const ENOSYS: ::c_int = 89;
pub const ENOTEMPTY: ::c_int = 93;
pub const ELOOP: ::c_int = 90;
pub const ENOMSG: ::c_int = 35;
pub const EIDRM: ::c_int = 36;
pub const ECHRNG: ::c_int = 37;
pub const EL2NSYNC: ::c_int = 38;
pub const EL3HLT: ::c_int = 39;
pub const EL3RST: ::c_int = 40;
pub const ELNRNG: ::c_int = 41;
pub const EUNATCH: ::c_int = 42;
pub const ENOCSI: ::c_int = 43;
pub const EL2HLT: ::c_int = 44;
pub const EBADE: ::c_int = 50;
pub const EBADR: ::c_int = 51;
pub const EXFULL: ::c_int = 52;
pub const ENOANO: ::c_int = 53;
pub const EBADRQC: ::c_int = 54;
pub const EBADSLT: ::c_int = 55;
pub const EDEADLOCK: ::c_int = 56;
pub const EMULTIHOP: ::c_int = 74;
pub const EOVERFLOW: ::c_int = 79;
pub const ENOTUNIQ: ::c_int = 80;
pub const EBADFD: ::c_int = 81;
pub const EBADMSG: ::c_int = 77;
pub const EREMCHG: ::c_int = 82;
pub const ELIBACC: ::c_int = 83;
pub const ELIBBAD: ::c_int = 84;
pub const ELIBSCN: ::c_int = 85;
pub const ELIBMAX: ::c_int = 86;
pub const ELIBEXEC: ::c_int = 87;
pub const EILSEQ: ::c_int = 88;
pub const ERESTART: ::c_int = 91;
pub const ESTRPIPE: ::c_int = 92;
pub const EUSERS: ::c_int = 94;
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
pub const EHOSTDOWN: ::c_int = 147;
pub const EHOSTUNREACH: ::c_int = 148;
pub const EALREADY: ::c_int = 149;
pub const EINPROGRESS: ::c_int = 150;
pub const ESTALE: ::c_int = 151;
pub const EUCLEAN: ::c_int = 135;
pub const ENOTNAM: ::c_int = 137;
pub const ENAVAIL: ::c_int = 138;
pub const EISNAM: ::c_int = 139;
pub const EREMOTEIO: ::c_int = 140;
pub const EDQUOT: ::c_int = 1133;
pub const ENOMEDIUM: ::c_int = 159;
pub const EMEDIUMTYPE: ::c_int = 160;
pub const ECANCELED: ::c_int = 158;
pub const ENOKEY: ::c_int = 161;
pub const EKEYEXPIRED: ::c_int = 162;
pub const EKEYREVOKED: ::c_int = 163;
pub const EKEYREJECTED: ::c_int = 164;
pub const EOWNERDEAD: ::c_int = 165;
pub const ENOTRECOVERABLE: ::c_int = 166;
pub const EHWPOISON: ::c_int = 168;
pub const ERFKILL: ::c_int = 167;

pub const SOCK_STREAM: ::c_int = 2;
pub const SOCK_DGRAM: ::c_int = 1;
pub const SOCK_SEQPACKET: ::c_int = 5;

pub const SOL_SOCKET: ::c_int = 65535;

pub const SO_REUSEADDR: ::c_int = 0x0004;
pub const SO_TYPE: ::c_int = 0x1008;
pub const SO_ERROR: ::c_int = 0x1007;
pub const SO_DONTROUTE: ::c_int = 0x0010;
pub const SO_BROADCAST: ::c_int = 0x0020;
pub const SO_SNDBUF: ::c_int = 0x1001;
pub const SO_RCVBUF: ::c_int = 0x1002;
pub const SO_KEEPALIVE: ::c_int = 0x0008;
pub const SO_OOBINLINE: ::c_int = 0x0100;
pub const SO_LINGER: ::c_int = 0x0080;
pub const SO_REUSEPORT: ::c_int = 0x200;
pub const SO_RCVLOWAT: ::c_int = 0x1004;
pub const SO_SNDLOWAT: ::c_int = 0x1003;
pub const SO_RCVTIMEO: ::c_int = 0x1006;
pub const SO_SNDTIMEO: ::c_int = 0x1005;
pub const SO_ACCEPTCONN: ::c_int = 0x1009;

pub const SA_ONSTACK: ::c_int = 0x08000000;
pub const SA_SIGINFO: ::c_int = 8;
pub const SA_NOCLDWAIT: ::c_int = 0x10000;

pub const SIGCHLD: ::c_int = 18;
pub const SIGBUS: ::c_int = 10;
pub const SIGTTIN: ::c_int = 26;
pub const SIGTTOU: ::c_int = 27;
pub const SIGXCPU: ::c_int = 30;
pub const SIGXFSZ: ::c_int = 31;
pub const SIGVTALRM: ::c_int = 28;
pub const SIGPROF: ::c_int = 29;
pub const SIGWINCH: ::c_int = 20;
pub const SIGUSR1: ::c_int = 16;
pub const SIGUSR2: ::c_int = 17;
pub const SIGCONT: ::c_int = 25;
pub const SIGSTOP: ::c_int = 23;
pub const SIGTSTP: ::c_int = 24;
pub const SIGURG: ::c_int = 21;
pub const SIGIO: ::c_int = 22;
pub const SIGSYS: ::c_int = 12;
pub const SIGSTKFLT: ::c_int = 7;
pub const SIGPOLL: ::c_int = ::SIGIO;
pub const SIGPWR: ::c_int = 19;
pub const SIG_SETMASK: ::c_int = 3;
pub const SIG_BLOCK: ::c_int = 1;
pub const SIG_UNBLOCK: ::c_int = 2;

pub const EXTPROC: ::tcflag_t = 0o200000;

pub const MAP_HUGETLB: ::c_int = 0x80000;

pub const F_GETLK: ::c_int = 33;
pub const F_GETOWN: ::c_int = 23;
pub const F_SETLK: ::c_int = 34;
pub const F_SETLKW: ::c_int = 35;
pub const F_SETOWN: ::c_int = 24;

pub const VEOF: usize = 16;
pub const VEOL: usize = 17;
pub const VEOL2: usize = 6;
pub const VMIN: usize = 4;
pub const IEXTEN: ::tcflag_t = 0o000400;
pub const TOSTOP: ::tcflag_t = 0o100000;
pub const FLUSHO: ::tcflag_t = 0o020000;

pub const TCGETS: ::c_int = 0x540D;
pub const TCSETS: ::c_int = 0x540E;
pub const TCSETSW: ::c_int = 0x540F;
pub const TCSETSF: ::c_int = 0x5410;
pub const TCGETA: ::c_int = 0x5401;
pub const TCSETA: ::c_int = 0x5402;
pub const TCSETAW: ::c_int = 0x5403;
pub const TCSETAF: ::c_int = 0x5404;
pub const TCSBRK: ::c_int = 0x5405;
pub const TCXONC: ::c_int = 0x5406;
pub const TCFLSH: ::c_int = 0x5407;
pub const TIOCGSOFTCAR: ::c_int = 0x5481;
pub const TIOCSSOFTCAR: ::c_int = 0x5482;
pub const TIOCLINUX: ::c_int = 0x5483;
pub const TIOCGSERIAL: ::c_int = 0x5484;
pub const TIOCEXCL: ::c_int = 0x740D;
pub const TIOCNXCL: ::c_int = 0x740E;
pub const TIOCSCTTY: ::c_int = 0x5480;
pub const TIOCGPGRP: ::c_int = 0x40047477;
pub const TIOCSPGRP: ::c_int = 0x80047476;
pub const TIOCOUTQ: ::c_int = 0x7472;
pub const TIOCSTI: ::c_int = 0x5472;
pub const TIOCGWINSZ: ::c_int = 0x40087468;
pub const TIOCSWINSZ: ::c_int = 0x80087467;
pub const TIOCMGET: ::c_int = 0x741D;
pub const TIOCMBIS: ::c_int = 0x741B;
pub const TIOCMBIC: ::c_int = 0x741C;
pub const TIOCMSET: ::c_int = 0x741A;
pub const FIONREAD: ::c_int = 0x467F;
pub const TIOCCONS: ::c_int = 0x80047478;

pub const SYS_gettid: ::c_long = 4222;   // Valid for O32
pub const SYS_perf_event_open: ::c_long = 4333;  // Valid for O32

pub const POLLWRNORM: ::c_short = 0x4;
pub const POLLWRBAND: ::c_short = 0x100;
