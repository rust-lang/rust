pub type clock_t = c_long;
pub type time_t = c_long;
pub type suseconds_t = c_long;
pub type ino_t = u64;
pub type off_t = i64;
pub type blkcnt_t = i64;

pub type blksize_t = c_long;
pub type fsblkcnt_t = ::c_ulonglong;
pub type fsfilcnt_t = ::c_ulonglong;
pub type rlim_t = ::c_ulonglong;

s! {
    pub struct sigaction {
        pub sa_sigaction: ::sighandler_t,
        pub sa_mask: ::sigset_t,
        pub sa_flags: ::c_int,
        _restorer: *mut ::c_void,
    }

    pub struct ipc_perm {
        pub __ipc_perm_key: ::key_t,
        pub uid: ::uid_t,
        pub gid: ::gid_t,
        pub cuid: ::uid_t,
        pub cgid: ::gid_t,
        pub mode: ::mode_t,
        pub __seq: ::c_int,
        __unused1: ::c_long,
        __unused2: ::c_long
    }

    pub struct termios {
        pub c_iflag: ::tcflag_t,
        pub c_oflag: ::tcflag_t,
        pub c_cflag: ::tcflag_t,
        pub c_lflag: ::tcflag_t,
        pub c_line: ::cc_t,
        pub c_cc: [::cc_t; ::NCCS],
        pub __c_ispeed: ::speed_t,
        pub __c_ospeed: ::speed_t,
    }

    pub struct flock {
        pub l_type: ::c_short,
        pub l_whence: ::c_short,
        pub l_start: ::off_t,
        pub l_len: ::off_t,
        pub l_pid: ::pid_t,
    }

    pub struct sysinfo {
        pub uptime: ::c_ulong,
        pub loads: [::c_ulong; 3],
        pub totalram: ::c_ulong,
        pub freeram: ::c_ulong,
        pub sharedram: ::c_ulong,
        pub bufferram: ::c_ulong,
        pub totalswap: ::c_ulong,
        pub freeswap: ::c_ulong,
        pub procs: ::c_ushort,
        pub pad: ::c_ushort,
        pub totalhigh: ::c_ulong,
        pub freehigh: ::c_ulong,
        pub mem_unit: ::c_uint,
        pub __reserved: [::c_char; 256],
    }
}

pub const BUFSIZ: ::c_uint = 1024;
pub const TMP_MAX: ::c_uint = 10000;
pub const FOPEN_MAX: ::c_uint = 1000;
pub const O_ACCMODE: ::c_int = 0o10000003;
pub const O_NDELAY: ::c_int = O_NONBLOCK;
pub const NI_MAXHOST: ::socklen_t = 255;
pub const PTHREAD_STACK_MIN: ::size_t = 2048;
pub const POSIX_FADV_DONTNEED: ::c_int = 4;
pub const POSIX_FADV_NOREUSE: ::c_int = 5;

pub const POSIX_MADV_DONTNEED: ::c_int = 4;

pub const RLIM_INFINITY: ::rlim_t = !0;
pub const RLIMIT_RTTIME: ::c_int = 15;
pub const RLIMIT_NLIMITS: ::c_int = 16;

pub const MAP_ANONYMOUS: ::c_int = MAP_ANON;

pub const TCP_COOKIE_TRANSACTIONS: ::c_int = 15;
pub const TCP_THIN_LINEAR_TIMEOUTS: ::c_int = 16;
pub const TCP_THIN_DUPACK: ::c_int = 17;
pub const TCP_USER_TIMEOUT: ::c_int = 18;
pub const TCP_REPAIR: ::c_int = 19;
pub const TCP_REPAIR_QUEUE: ::c_int = 20;
pub const TCP_QUEUE_SEQ: ::c_int = 21;
pub const TCP_REPAIR_OPTIONS: ::c_int = 22;
pub const TCP_FASTOPEN: ::c_int = 23;
pub const TCP_TIMESTAMP: ::c_int = 24;

pub const SIGUNUSED: ::c_int = ::SIGSYS;

pub const FALLOC_FL_KEEP_SIZE: ::c_int = 0x01;
pub const FALLOC_FL_PUNCH_HOLE: ::c_int = 0x02;

pub const __SIZEOF_PTHREAD_CONDATTR_T: usize = 4;
pub const __SIZEOF_PTHREAD_MUTEXATTR_T: usize = 4;

pub const CPU_SETSIZE: ::c_int = 128;

pub const QFMT_VFS_V1: ::c_int = 4;

pub const PTRACE_TRACEME: ::c_int = 0;
pub const PTRACE_PEEKTEXT: ::c_int = 1;
pub const PTRACE_PEEKDATA: ::c_int = 2;
pub const PTRACE_PEEKUSER: ::c_int = 3;
pub const PTRACE_POKETEXT: ::c_int = 4;
pub const PTRACE_POKEDATA: ::c_int = 5;
pub const PTRACE_POKEUSER: ::c_int = 6;
pub const PTRACE_CONT: ::c_int = 7;
pub const PTRACE_KILL: ::c_int = 8;
pub const PTRACE_SINGLESTEP: ::c_int = 9;
pub const PTRACE_ATTACH: ::c_int = 16;
pub const PTRACE_DETACH: ::c_int = 17;
pub const PTRACE_SYSCALL: ::c_int = 24;
pub const PTRACE_SETOPTIONS: ::c_int = 0x4200;
pub const PTRACE_GETEVENTMSG: ::c_int = 0x4201;
pub const PTRACE_GETSIGINFO: ::c_int = 0x4202;
pub const PTRACE_SETSIGINFO: ::c_int = 0x4203;
pub const PTRACE_GETREGSET: ::c_int = 0x4204;
pub const PTRACE_SETREGSET: ::c_int = 0x4205;
pub const PTRACE_SEIZE: ::c_int = 0x4206;
pub const PTRACE_INTERRUPT: ::c_int = 0x4207;
pub const PTRACE_LISTEN: ::c_int = 0x4208;
pub const PTRACE_PEEKSIGINFO: ::c_int = 0x4209;

pub const PTRACE_O_EXITKILL: ::c_int = 1048576;
pub const PTRACE_O_TRACECLONE: ::c_int = 8;
pub const PTRACE_O_TRACEEXEC: ::c_int = 16;
pub const PTRACE_O_TRACEEXIT: ::c_int = 64;
pub const PTRACE_O_TRACEFORK: ::c_int = 2;
pub const PTRACE_O_TRACESYSGOOD: ::c_int = 1;
pub const PTRACE_O_TRACEVFORK: ::c_int = 4;
pub const PTRACE_O_TRACEVFORKDONE: ::c_int = 32;
pub const PTRACE_O_SUSPEND_SECCOMP: ::c_int = 2097152;

pub const MADV_DODUMP: ::c_int = 17;
pub const MADV_DONTDUMP: ::c_int = 16;

pub const EPOLLWAKEUP: ::c_int = 0x20000000;

pub const POLLRDNORM: ::c_short = 0x040;
pub const POLLRDBAND: ::c_short = 0x080;

pub const MADV_HUGEPAGE: ::c_int = 14;
pub const MADV_NOHUGEPAGE: ::c_int = 15;

pub const PTRACE_GETFPREGS: ::c_uint = 14;
pub const PTRACE_SETFPREGS: ::c_uint = 15;
pub const PTRACE_GETFPXREGS: ::c_uint = 18;
pub const PTRACE_SETFPXREGS: ::c_uint = 19;
pub const PTRACE_GETREGS: ::c_uint = 12;
pub const PTRACE_SETREGS: ::c_uint = 13;

pub const EFD_NONBLOCK: ::c_int = ::O_NONBLOCK;

pub const SFD_NONBLOCK: ::c_int = ::O_NONBLOCK;

pub const TCSANOW: ::c_int = 0;
pub const TCSADRAIN: ::c_int = 1;
pub const TCSAFLUSH: ::c_int = 2;

pub const TIOCINQ: ::c_int = ::FIONREAD;

pub const RTLD_GLOBAL: ::c_int = 0x100;
pub const RTLD_NOLOAD: ::c_int = 0x4;

// TODO(#247) Temporarily musl-specific (available since musl 0.9.12 / Linux
// kernel 3.10).  See also notbsd/mod.rs
pub const CLOCK_SGI_CYCLE: ::clockid_t = 10;
pub const CLOCK_TAI: ::clockid_t = 11;

pub const MCL_CURRENT: ::c_int = 0x0001;
pub const MCL_FUTURE: ::c_int = 0x0002;

pub const SIGSTKSZ: ::size_t = 8192;
pub const CBAUD: ::tcflag_t = 0o0010017;
pub const TAB1: ::c_int = 0x00000800;
pub const TAB2: ::c_int = 0x00001000;
pub const TAB3: ::c_int = 0x00001800;
pub const CR1: ::c_int  = 0x00000200;
pub const CR2: ::c_int  = 0x00000400;
pub const CR3: ::c_int  = 0x00000600;
pub const FF1: ::c_int  = 0x00008000;
pub const BS1: ::c_int  = 0x00002000;
pub const VT1: ::c_int  = 0x00004000;
pub const VWERASE: usize = 14;
pub const VREPRINT: usize = 12;
pub const VSUSP: usize = 10;
pub const VSTART: usize = 8;
pub const VSTOP: usize = 9;
pub const VDISCARD: usize = 13;
pub const VTIME: usize = 5;
pub const IXON: ::tcflag_t = 0x00000400;
pub const IXOFF: ::tcflag_t = 0x00001000;
pub const ONLCR: ::tcflag_t = 0x4;
pub const CSIZE: ::tcflag_t = 0x00000030;
pub const CS6: ::tcflag_t = 0x00000010;
pub const CS7: ::tcflag_t = 0x00000020;
pub const CS8: ::tcflag_t = 0x00000030;
pub const CSTOPB: ::tcflag_t = 0x00000040;
pub const CREAD: ::tcflag_t = 0x00000080;
pub const PARENB: ::tcflag_t = 0x00000100;
pub const PARODD: ::tcflag_t = 0x00000200;
pub const HUPCL: ::tcflag_t = 0x00000400;
pub const CLOCAL: ::tcflag_t = 0x00000800;
pub const ECHOKE: ::tcflag_t = 0x00000800;
pub const ECHOE: ::tcflag_t = 0x00000010;
pub const ECHOK: ::tcflag_t = 0x00000020;
pub const ECHONL: ::tcflag_t = 0x00000040;
pub const ECHOPRT: ::tcflag_t = 0x00000400;
pub const ECHOCTL: ::tcflag_t = 0x00000200;
pub const ISIG: ::tcflag_t = 0x00000001;
pub const ICANON: ::tcflag_t = 0x00000002;
pub const PENDIN: ::tcflag_t = 0x00004000;
pub const NOFLSH: ::tcflag_t = 0x00000080;

extern {
    pub fn ioctl(fd: ::c_int, request: ::c_int, ...) -> ::c_int;
    pub fn ptrace(request: ::c_int, ...) -> ::c_long;
    pub fn getpriority(which: ::c_int, who: ::id_t) -> ::c_int;
    pub fn setpriority(which: ::c_int, who: ::id_t, prio: ::c_int) -> ::c_int;
}

cfg_if! {
    if #[cfg(any(target_arch = "x86_64"))] {
        mod b64;
        pub use self::b64::*;
    } else if #[cfg(any(target_arch = "x86",
                        target_arch = "mips",
                        target_arch = "arm",
                        target_arch = "asmjs",
                        target_arch = "wasm32"))] {
        mod b32;
        pub use self::b32::*;
    } else { }
}
