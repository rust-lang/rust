//! PowerPC64-specific definitions for 64-bit linux-like values

pub type c_char = u8;
pub type wchar_t = i32;
pub type nlink_t = u64;
pub type blksize_t = i64;

s! {
    pub struct stat {
        pub st_dev: ::dev_t,
        pub st_ino: ::ino_t,
        pub st_nlink: ::nlink_t,
        pub st_mode: ::mode_t,
        pub st_uid: ::uid_t,
        pub st_gid: ::gid_t,
        __pad0: ::c_int,
        pub st_rdev: ::dev_t,
        pub st_size: ::off_t,
        pub st_blksize: ::blksize_t,
        pub st_blocks: ::blkcnt_t,
        pub st_atime: ::time_t,
        pub st_atime_nsec: ::c_long,
        pub st_mtime: ::time_t,
        pub st_mtime_nsec: ::c_long,
        pub st_ctime: ::time_t,
        pub st_ctime_nsec: ::c_long,
        __unused: [::c_long; 3],
    }

    pub struct stat64 {
        pub st_dev: ::dev_t,
        pub st_ino: ::ino64_t,
        pub st_nlink: ::nlink_t,
        pub st_mode: ::mode_t,
        pub st_uid: ::uid_t,
        pub st_gid: ::gid_t,
        __pad0: ::c_int,
        pub st_rdev: ::dev_t,
        pub st_size: ::off64_t,
        pub st_blksize: ::blksize_t,
        pub st_blocks: ::blkcnt64_t,
        pub st_atime: ::time_t,
        pub st_atime_nsec: ::c_long,
        pub st_mtime: ::time_t,
        pub st_mtime_nsec: ::c_long,
        pub st_ctime: ::time_t,
        pub st_ctime_nsec: ::c_long,
        __reserved: [::c_long; 3],
    }

    pub struct pthread_attr_t {
        __size: [u64; 7]
    }

    pub struct ipc_perm {
        pub __key: ::key_t,
        pub uid: ::uid_t,
        pub gid: ::gid_t,
        pub cuid: ::uid_t,
        pub cgid: ::gid_t,
        pub mode: ::mode_t,
        pub __seq: ::uint32_t,
        __pad1: ::uint32_t,
        __unused1: ::uint64_t,
        __unused2: ::c_ulong,
    }

    pub struct shmid_ds {
        pub shm_perm: ::ipc_perm,
        pub shm_atime: ::time_t,
        pub shm_dtime: ::time_t,
        pub shm_ctime: ::time_t,
        pub shm_segsz: ::size_t,
        pub shm_cpid: ::pid_t,
        pub shm_lpid: ::pid_t,
        pub shm_nattch: ::shmatt_t,
        __unused4: ::c_ulong,
        __unused5: ::c_ulong
    }
}

pub const __SIZEOF_PTHREAD_CONDATTR_T: usize = 4;
pub const __SIZEOF_PTHREAD_MUTEX_T: usize = 40;
pub const __SIZEOF_PTHREAD_MUTEXATTR_T: usize = 4;

pub const O_DIRECTORY: ::c_int = 0x4000;
pub const O_NOFOLLOW: ::c_int = 0x8000;
pub const O_DIRECT: ::c_int = 0x20000;

pub const MAP_LOCKED: ::c_int = 0x00080;
pub const MAP_NORESERVE: ::c_int = 0x00040;

pub const EDEADLOCK: ::c_int = 58;

pub const SO_PEERCRED: ::c_int = 21;
pub const SO_RCVLOWAT: ::c_int = 16;
pub const SO_SNDLOWAT: ::c_int = 17;
pub const SO_RCVTIMEO: ::c_int = 18;
pub const SO_SNDTIMEO: ::c_int = 19;

pub const FIOCLEX: ::c_ulong = 0x20006601;
pub const FIONBIO: ::c_ulong = 0x8004667e;

pub const SYS_gettid: ::c_long = 207;
pub const SYS_perf_event_open: ::c_long = 319;

pub const MCL_CURRENT: ::c_int = 0x2000;
pub const MCL_FUTURE: ::c_int = 0x4000;

pub const SIGSTKSZ: ::size_t = 0x4000;
pub const CBAUD: ::tcflag_t = 0xff;
pub const TAB1: ::c_int = 0x400;
pub const TAB2: ::c_int = 0x800;
pub const TAB3: ::c_int = 0xc00;
pub const CR1: ::c_int  = 0x1000;
pub const CR2: ::c_int  = 0x2000;
pub const CR3: ::c_int  = 0x3000;
pub const FF1: ::c_int  = 0x4000;
pub const BS1: ::c_int  = 0x8000;
pub const VT1: ::c_int  = 0x10000;
pub const VWERASE: usize = 0xa;
pub const VREPRINT: usize = 0xb;
pub const VSUSP: usize = 0xc;
pub const VSTART: usize = 0xd;
pub const VSTOP: usize = 0xe;
pub const VDISCARD: usize = 0x10;
pub const VTIME: usize = 0x7;
pub const IXON: ::tcflag_t = 0x200;
pub const IXOFF: ::tcflag_t = 0x400;
pub const ONLCR: ::tcflag_t = 0x2;
pub const CSIZE: ::tcflag_t = 0x300;
pub const CS6: ::tcflag_t = 0x100;
pub const CS7: ::tcflag_t = 0x200;
pub const CS8: ::tcflag_t = 0x300;
pub const CSTOPB: ::tcflag_t = 0x400;
pub const CREAD: ::tcflag_t = 0x800;
pub const PARENB: ::tcflag_t = 0x1000;
pub const PARODD: ::tcflag_t = 0x2000;
pub const HUPCL: ::tcflag_t = 0x4000;
pub const CLOCAL: ::tcflag_t = 0x8000;
pub const ECHOKE: ::tcflag_t = 0x1;
pub const ECHOE: ::tcflag_t = 0x2;
pub const ECHOK: ::tcflag_t = 0x4;
pub const ECHONL: ::tcflag_t = 0x10;
pub const ECHOPRT: ::tcflag_t = 0x20;
pub const ECHOCTL: ::tcflag_t = 0x40;
pub const ISIG: ::tcflag_t = 0x80;
pub const ICANON: ::tcflag_t = 0x100;
pub const PENDIN: ::tcflag_t = 0x20000000;
pub const NOFLSH: ::tcflag_t = 0x80000000;

pub const VEOL: usize = 6;
pub const VEOL2: usize = 8;
pub const VMIN: usize = 5;
pub const IEXTEN: ::tcflag_t = 0x400;
pub const TOSTOP: ::tcflag_t = 0x400000;
pub const FLUSHO: ::tcflag_t = 0x800000;
pub const EXTPROC: ::tcflag_t = 0x10000000;
pub const TCGETS: ::c_ulong = 0x403c7413;
pub const TCSETS: ::c_ulong = 0x803c7414;
pub const TCSETSW: ::c_ulong = 0x803c7415;
pub const TCSETSF: ::c_ulong = 0x803c7416;
pub const TCGETA: ::c_ulong = 0x40147417;
pub const TCSETA: ::c_ulong = 0x80147418;
pub const TCSETAW: ::c_ulong = 0x80147419;
pub const TCSETAF: ::c_ulong = 0x8014741c;
pub const TCSBRK: ::c_ulong = 0x2000741d;
pub const TCXONC: ::c_ulong = 0x2000741e;
pub const TCFLSH: ::c_ulong = 0x2000741f;
pub const TIOCINQ: ::c_ulong = 0x4004667f;
pub const TIOCGPGRP: ::c_ulong = 0x40047477;
pub const TIOCSPGRP: ::c_ulong = 0x80047476;
pub const TIOCOUTQ: ::c_ulong = 0x40047473;
pub const TIOCGWINSZ: ::c_ulong = 0x40087468;
pub const TIOCSWINSZ: ::c_ulong = 0x80087467;
pub const FIONREAD: ::c_ulong = 0x4004667f;
