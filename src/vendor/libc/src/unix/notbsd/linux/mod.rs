//! Linux-specific definitions for linux-like values

use dox::mem;

pub type useconds_t = u32;
pub type dev_t = u64;
pub type socklen_t = u32;
pub type pthread_t = c_ulong;
pub type mode_t = u32;
pub type ino64_t = u64;
pub type off64_t = i64;
pub type blkcnt64_t = i64;
pub type rlim64_t = u64;
pub type shmatt_t = ::c_ulong;
pub type mqd_t = ::c_int;
pub type msgqnum_t = ::c_ulong;
pub type msglen_t = ::c_ulong;
pub type nfds_t = ::c_ulong;
pub type nl_item = ::c_int;

pub enum fpos64_t {} // TODO: fill this out with a struct

s! {
    pub struct dirent {
        pub d_ino: ::ino_t,
        pub d_off: ::off_t,
        pub d_reclen: ::c_ushort,
        pub d_type: ::c_uchar,
        pub d_name: [::c_char; 256],
    }

    pub struct dirent64 {
        pub d_ino: ::ino64_t,
        pub d_off: ::off64_t,
        pub d_reclen: ::c_ushort,
        pub d_type: ::c_uchar,
        pub d_name: [::c_char; 256],
    }

    pub struct rlimit64 {
        pub rlim_cur: rlim64_t,
        pub rlim_max: rlim64_t,
    }

    pub struct glob_t {
        pub gl_pathc: ::size_t,
        pub gl_pathv: *mut *mut c_char,
        pub gl_offs: ::size_t,
        pub gl_flags: ::c_int,

        __unused1: *mut ::c_void,
        __unused2: *mut ::c_void,
        __unused3: *mut ::c_void,
        __unused4: *mut ::c_void,
        __unused5: *mut ::c_void,
    }

    pub struct ifaddrs {
        pub ifa_next: *mut ifaddrs,
        pub ifa_name: *mut c_char,
        pub ifa_flags: ::c_uint,
        pub ifa_addr: *mut ::sockaddr,
        pub ifa_netmask: *mut ::sockaddr,
        pub ifa_ifu: *mut ::sockaddr, // FIXME This should be a union
        pub ifa_data: *mut ::c_void
    }

    pub struct pthread_mutex_t {
        #[cfg(any(target_arch = "mips", target_arch = "mipsel",
                  target_arch = "arm", target_arch = "powerpc"))]
        __align: [::c_long; 0],
        #[cfg(not(any(target_arch = "mips", target_arch = "mipsel",
                      target_arch = "arm", target_arch = "powerpc")))]
        __align: [::c_longlong; 0],
        size: [u8; __SIZEOF_PTHREAD_MUTEX_T],
    }

    pub struct pthread_rwlock_t {
        #[cfg(any(target_arch = "mips", target_arch = "mipsel",
                  target_arch = "arm", target_arch = "powerpc"))]
        __align: [::c_long; 0],
        #[cfg(not(any(target_arch = "mips", target_arch = "mipsel",
                      target_arch = "arm", target_arch = "powerpc")))]
        __align: [::c_longlong; 0],
        size: [u8; __SIZEOF_PTHREAD_RWLOCK_T],
    }

    pub struct pthread_mutexattr_t {
        #[cfg(any(target_arch = "x86_64", target_arch = "powerpc64",
                  target_arch = "mips64", target_arch = "s390x"))]
        __align: [::c_int; 0],
        #[cfg(not(any(target_arch = "x86_64", target_arch = "powerpc64",
                      target_arch = "mips64", target_arch = "s390x")))]
        __align: [::c_long; 0],
        size: [u8; __SIZEOF_PTHREAD_MUTEXATTR_T],
    }

    pub struct pthread_cond_t {
        #[cfg(any(target_env = "musl"))]
        __align: [*const ::c_void; 0],
        #[cfg(not(any(target_env = "musl")))]
        __align: [::c_longlong; 0],
        size: [u8; __SIZEOF_PTHREAD_COND_T],
    }

    pub struct pthread_condattr_t {
        __align: [::c_int; 0],
        size: [u8; __SIZEOF_PTHREAD_CONDATTR_T],
    }

    pub struct passwd {
        pub pw_name: *mut ::c_char,
        pub pw_passwd: *mut ::c_char,
        pub pw_uid: ::uid_t,
        pub pw_gid: ::gid_t,
        pub pw_gecos: *mut ::c_char,
        pub pw_dir: *mut ::c_char,
        pub pw_shell: *mut ::c_char,
    }

    pub struct spwd {
        pub sp_namp: *mut ::c_char,
        pub sp_pwdp: *mut ::c_char,
        pub sp_lstchg: ::c_long,
        pub sp_min: ::c_long,
        pub sp_max: ::c_long,
        pub sp_warn: ::c_long,
        pub sp_inact: ::c_long,
        pub sp_expire: ::c_long,
        pub sp_flag: ::c_ulong,
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
        #[cfg(target_endian = "little")]
        pub f_fsid: ::c_ulong,
        #[cfg(target_pointer_width = "32")]
        __f_unused: ::c_int,
        #[cfg(target_endian = "big")]
        pub f_fsid: ::c_ulong,
        pub f_flag: ::c_ulong,
        pub f_namemax: ::c_ulong,
        __f_spare: [::c_int; 6],
    }

    pub struct dqblk {
        pub dqb_bhardlimit: ::uint64_t,
        pub dqb_bsoftlimit: ::uint64_t,
        pub dqb_curspace: ::uint64_t,
        pub dqb_ihardlimit: ::uint64_t,
        pub dqb_isoftlimit: ::uint64_t,
        pub dqb_curinodes: ::uint64_t,
        pub dqb_btime: ::uint64_t,
        pub dqb_itime: ::uint64_t,
        pub dqb_valid: ::uint32_t,
    }

    pub struct signalfd_siginfo {
        pub ssi_signo: ::uint32_t,
        pub ssi_errno: ::int32_t,
        pub ssi_code: ::int32_t,
        pub ssi_pid: ::uint32_t,
        pub ssi_uid: ::uint32_t,
        pub ssi_fd: ::int32_t,
        pub ssi_tid: ::uint32_t,
        pub ssi_band: ::uint32_t,
        pub ssi_overrun: ::uint32_t,
        pub ssi_trapno: ::uint32_t,
        pub ssi_status: ::int32_t,
        pub ssi_int: ::int32_t,
        pub ssi_ptr: ::uint64_t,
        pub ssi_utime: ::uint64_t,
        pub ssi_stime: ::uint64_t,
        pub ssi_addr: ::uint64_t,
        _pad: [::uint8_t; 48],
    }

    pub struct fsid_t {
        __val: [::c_int; 2],
    }

    pub struct mq_attr {
        pub mq_flags: ::c_long,
        pub mq_maxmsg: ::c_long,
        pub mq_msgsize: ::c_long,
        pub mq_curmsgs: ::c_long,
        pad: [::c_long; 4]
    }

    pub struct cpu_set_t {
        #[cfg(target_pointer_width = "32")]
        bits: [u32; 32],
        #[cfg(target_pointer_width = "64")]
        bits: [u64; 16],
    }

    pub struct if_nameindex {
        pub if_index: ::c_uint,
        pub if_name: *mut ::c_char,
    }

    // System V IPC
    pub struct msginfo {
        pub msgpool: ::c_int,
        pub msgmap: ::c_int,
        pub msgmax: ::c_int,
        pub msgmnb: ::c_int,
        pub msgmni: ::c_int,
        pub msgssz: ::c_int,
        pub msgtql: ::c_int,
        pub msgseg: ::c_ushort,
    }
}

pub const ABDAY_1: ::nl_item = 0x20000;
pub const ABDAY_2: ::nl_item = 0x20001;
pub const ABDAY_3: ::nl_item = 0x20002;
pub const ABDAY_4: ::nl_item = 0x20003;
pub const ABDAY_5: ::nl_item = 0x20004;
pub const ABDAY_6: ::nl_item = 0x20005;
pub const ABDAY_7: ::nl_item = 0x20006;

pub const DAY_1: ::nl_item = 0x20007;
pub const DAY_2: ::nl_item = 0x20008;
pub const DAY_3: ::nl_item = 0x20009;
pub const DAY_4: ::nl_item = 0x2000A;
pub const DAY_5: ::nl_item = 0x2000B;
pub const DAY_6: ::nl_item = 0x2000C;
pub const DAY_7: ::nl_item = 0x2000D;

pub const ABMON_1: ::nl_item = 0x2000E;
pub const ABMON_2: ::nl_item = 0x2000F;
pub const ABMON_3: ::nl_item = 0x20010;
pub const ABMON_4: ::nl_item = 0x20011;
pub const ABMON_5: ::nl_item = 0x20012;
pub const ABMON_6: ::nl_item = 0x20013;
pub const ABMON_7: ::nl_item = 0x20014;
pub const ABMON_8: ::nl_item = 0x20015;
pub const ABMON_9: ::nl_item = 0x20016;
pub const ABMON_10: ::nl_item = 0x20017;
pub const ABMON_11: ::nl_item = 0x20018;
pub const ABMON_12: ::nl_item = 0x20019;

pub const CLONE_NEWCGROUP: ::c_int = 0x02000000;

pub const MON_1: ::nl_item = 0x2001A;
pub const MON_2: ::nl_item = 0x2001B;
pub const MON_3: ::nl_item = 0x2001C;
pub const MON_4: ::nl_item = 0x2001D;
pub const MON_5: ::nl_item = 0x2001E;
pub const MON_6: ::nl_item = 0x2001F;
pub const MON_7: ::nl_item = 0x20020;
pub const MON_8: ::nl_item = 0x20021;
pub const MON_9: ::nl_item = 0x20022;
pub const MON_10: ::nl_item = 0x20023;
pub const MON_11: ::nl_item = 0x20024;
pub const MON_12: ::nl_item = 0x20025;

pub const AM_STR: ::nl_item = 0x20026;
pub const PM_STR: ::nl_item = 0x20027;

pub const D_T_FMT: ::nl_item = 0x20028;
pub const D_FMT: ::nl_item = 0x20029;
pub const T_FMT: ::nl_item = 0x2002A;
pub const T_FMT_AMPM: ::nl_item = 0x2002B;

pub const ERA: ::nl_item = 0x2002C;
pub const ERA_D_FMT: ::nl_item = 0x2002E;
pub const ALT_DIGITS: ::nl_item = 0x2002F;
pub const ERA_D_T_FMT: ::nl_item = 0x20030;
pub const ERA_T_FMT: ::nl_item = 0x20031;

pub const CODESET: ::nl_item = 14;

pub const CRNCYSTR: ::nl_item = 0x4000F;

pub const RUSAGE_THREAD: ::c_int = 1;
pub const RUSAGE_CHILDREN: ::c_int = -1;

pub const RADIXCHAR: ::nl_item = 0x10000;
pub const THOUSEP: ::nl_item = 0x10001;

pub const YESEXPR: ::nl_item = 0x50000;
pub const NOEXPR: ::nl_item = 0x50001;
pub const YESSTR: ::nl_item = 0x50002;
pub const NOSTR: ::nl_item = 0x50003;

pub const FILENAME_MAX: ::c_uint = 4096;
pub const L_tmpnam: ::c_uint = 20;
pub const _PC_LINK_MAX: ::c_int = 0;
pub const _PC_MAX_CANON: ::c_int = 1;
pub const _PC_MAX_INPUT: ::c_int = 2;
pub const _PC_NAME_MAX: ::c_int = 3;
pub const _PC_PATH_MAX: ::c_int = 4;
pub const _PC_PIPE_BUF: ::c_int = 5;
pub const _PC_CHOWN_RESTRICTED: ::c_int = 6;
pub const _PC_NO_TRUNC: ::c_int = 7;
pub const _PC_VDISABLE: ::c_int = 8;

pub const _SC_ARG_MAX: ::c_int = 0;
pub const _SC_CHILD_MAX: ::c_int = 1;
pub const _SC_CLK_TCK: ::c_int = 2;
pub const _SC_NGROUPS_MAX: ::c_int = 3;
pub const _SC_OPEN_MAX: ::c_int = 4;
pub const _SC_STREAM_MAX: ::c_int = 5;
pub const _SC_TZNAME_MAX: ::c_int = 6;
pub const _SC_JOB_CONTROL: ::c_int = 7;
pub const _SC_SAVED_IDS: ::c_int = 8;
pub const _SC_REALTIME_SIGNALS: ::c_int = 9;
pub const _SC_PRIORITY_SCHEDULING: ::c_int = 10;
pub const _SC_TIMERS: ::c_int = 11;
pub const _SC_ASYNCHRONOUS_IO: ::c_int = 12;
pub const _SC_PRIORITIZED_IO: ::c_int = 13;
pub const _SC_SYNCHRONIZED_IO: ::c_int = 14;
pub const _SC_FSYNC: ::c_int = 15;
pub const _SC_MAPPED_FILES: ::c_int = 16;
pub const _SC_MEMLOCK: ::c_int = 17;
pub const _SC_MEMLOCK_RANGE: ::c_int = 18;
pub const _SC_MEMORY_PROTECTION: ::c_int = 19;
pub const _SC_MESSAGE_PASSING: ::c_int = 20;
pub const _SC_SEMAPHORES: ::c_int = 21;
pub const _SC_SHARED_MEMORY_OBJECTS: ::c_int = 22;
pub const _SC_AIO_LISTIO_MAX: ::c_int = 23;
pub const _SC_AIO_MAX: ::c_int = 24;
pub const _SC_AIO_PRIO_DELTA_MAX: ::c_int = 25;
pub const _SC_DELAYTIMER_MAX: ::c_int = 26;
pub const _SC_MQ_OPEN_MAX: ::c_int = 27;
pub const _SC_MQ_PRIO_MAX: ::c_int = 28;
pub const _SC_VERSION: ::c_int = 29;
pub const _SC_PAGESIZE: ::c_int = 30;
pub const _SC_PAGE_SIZE: ::c_int = _SC_PAGESIZE;
pub const _SC_RTSIG_MAX: ::c_int = 31;
pub const _SC_SEM_NSEMS_MAX: ::c_int = 32;
pub const _SC_SEM_VALUE_MAX: ::c_int = 33;
pub const _SC_SIGQUEUE_MAX: ::c_int = 34;
pub const _SC_TIMER_MAX: ::c_int = 35;
pub const _SC_BC_BASE_MAX: ::c_int = 36;
pub const _SC_BC_DIM_MAX: ::c_int = 37;
pub const _SC_BC_SCALE_MAX: ::c_int = 38;
pub const _SC_BC_STRING_MAX: ::c_int = 39;
pub const _SC_COLL_WEIGHTS_MAX: ::c_int = 40;
pub const _SC_EXPR_NEST_MAX: ::c_int = 42;
pub const _SC_LINE_MAX: ::c_int = 43;
pub const _SC_RE_DUP_MAX: ::c_int = 44;
pub const _SC_2_VERSION: ::c_int = 46;
pub const _SC_2_C_BIND: ::c_int = 47;
pub const _SC_2_C_DEV: ::c_int = 48;
pub const _SC_2_FORT_DEV: ::c_int = 49;
pub const _SC_2_FORT_RUN: ::c_int = 50;
pub const _SC_2_SW_DEV: ::c_int = 51;
pub const _SC_2_LOCALEDEF: ::c_int = 52;
pub const _SC_IOV_MAX: ::c_int = 60;
pub const _SC_THREADS: ::c_int = 67;
pub const _SC_THREAD_SAFE_FUNCTIONS: ::c_int = 68;
pub const _SC_GETGR_R_SIZE_MAX: ::c_int = 69;
pub const _SC_GETPW_R_SIZE_MAX: ::c_int = 70;
pub const _SC_LOGIN_NAME_MAX: ::c_int = 71;
pub const _SC_TTY_NAME_MAX: ::c_int = 72;
pub const _SC_THREAD_DESTRUCTOR_ITERATIONS: ::c_int = 73;
pub const _SC_THREAD_KEYS_MAX: ::c_int = 74;
pub const _SC_THREAD_STACK_MIN: ::c_int = 75;
pub const _SC_THREAD_THREADS_MAX: ::c_int = 76;
pub const _SC_THREAD_ATTR_STACKADDR: ::c_int = 77;
pub const _SC_THREAD_ATTR_STACKSIZE: ::c_int = 78;
pub const _SC_THREAD_PRIORITY_SCHEDULING: ::c_int = 79;
pub const _SC_THREAD_PRIO_INHERIT: ::c_int = 80;
pub const _SC_THREAD_PRIO_PROTECT: ::c_int = 81;
pub const _SC_NPROCESSORS_ONLN: ::c_int = 84;
pub const _SC_ATEXIT_MAX: ::c_int = 87;
pub const _SC_XOPEN_VERSION: ::c_int = 89;
pub const _SC_XOPEN_XCU_VERSION: ::c_int = 90;
pub const _SC_XOPEN_UNIX: ::c_int = 91;
pub const _SC_XOPEN_CRYPT: ::c_int = 92;
pub const _SC_XOPEN_ENH_I18N: ::c_int = 93;
pub const _SC_XOPEN_SHM: ::c_int = 94;
pub const _SC_2_CHAR_TERM: ::c_int = 95;
pub const _SC_2_UPE: ::c_int = 97;
pub const _SC_XBS5_ILP32_OFF32: ::c_int = 125;
pub const _SC_XBS5_ILP32_OFFBIG: ::c_int = 126;
pub const _SC_XBS5_LPBIG_OFFBIG: ::c_int = 128;
pub const _SC_XOPEN_LEGACY: ::c_int = 129;
pub const _SC_XOPEN_REALTIME: ::c_int = 130;
pub const _SC_XOPEN_REALTIME_THREADS: ::c_int = 131;
pub const _SC_HOST_NAME_MAX: ::c_int = 180;

pub const RLIM_SAVED_MAX: ::rlim_t = RLIM_INFINITY;
pub const RLIM_SAVED_CUR: ::rlim_t = RLIM_INFINITY;

pub const GLOB_ERR: ::c_int = 1 << 0;
pub const GLOB_MARK: ::c_int = 1 << 1;
pub const GLOB_NOSORT: ::c_int = 1 << 2;
pub const GLOB_DOOFFS: ::c_int = 1 << 3;
pub const GLOB_NOCHECK: ::c_int = 1 << 4;
pub const GLOB_APPEND: ::c_int = 1 << 5;
pub const GLOB_NOESCAPE: ::c_int = 1 << 6;

pub const GLOB_NOSPACE: ::c_int = 1;
pub const GLOB_ABORTED: ::c_int = 2;
pub const GLOB_NOMATCH: ::c_int = 3;

pub const POSIX_MADV_NORMAL: ::c_int = 0;
pub const POSIX_MADV_RANDOM: ::c_int = 1;
pub const POSIX_MADV_SEQUENTIAL: ::c_int = 2;
pub const POSIX_MADV_WILLNEED: ::c_int = 3;

pub const S_IEXEC: mode_t = 64;
pub const S_IWRITE: mode_t = 128;
pub const S_IREAD: mode_t = 256;

pub const F_LOCK: ::c_int = 1;
pub const F_TEST: ::c_int = 3;
pub const F_TLOCK: ::c_int = 2;
pub const F_ULOCK: ::c_int = 0;

pub const ST_RDONLY: ::c_ulong = 1;
pub const ST_NOSUID: ::c_ulong = 2;
pub const ST_NODEV: ::c_ulong = 4;
pub const ST_NOEXEC: ::c_ulong = 8;
pub const ST_SYNCHRONOUS: ::c_ulong = 16;
pub const ST_MANDLOCK: ::c_ulong = 64;
pub const ST_WRITE: ::c_ulong = 128;
pub const ST_APPEND: ::c_ulong = 256;
pub const ST_IMMUTABLE: ::c_ulong = 512;
pub const ST_NOATIME: ::c_ulong = 1024;
pub const ST_NODIRATIME: ::c_ulong = 2048;

pub const RTLD_NEXT: *mut ::c_void = -1i64 as *mut ::c_void;
pub const RTLD_DEFAULT: *mut ::c_void = 0i64 as *mut ::c_void;
pub const RTLD_NODELETE: ::c_int = 0x1000;
pub const RTLD_NOW: ::c_int = 0x2;

pub const TCP_MD5SIG: ::c_int = 14;

pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
    __align: [],
    size: [0; __SIZEOF_PTHREAD_MUTEX_T],
};
pub const PTHREAD_COND_INITIALIZER: pthread_cond_t = pthread_cond_t {
    __align: [],
    size: [0; __SIZEOF_PTHREAD_COND_T],
};
pub const PTHREAD_RWLOCK_INITIALIZER: pthread_rwlock_t = pthread_rwlock_t {
    __align: [],
    size: [0; __SIZEOF_PTHREAD_RWLOCK_T],
};
pub const PTHREAD_MUTEX_NORMAL: ::c_int = 0;
pub const PTHREAD_MUTEX_RECURSIVE: ::c_int = 1;
pub const PTHREAD_MUTEX_ERRORCHECK: ::c_int = 2;
pub const PTHREAD_MUTEX_DEFAULT: ::c_int = PTHREAD_MUTEX_NORMAL;
pub const __SIZEOF_PTHREAD_COND_T: usize = 48;

pub const SCHED_OTHER: ::c_int = 0;
pub const SCHED_FIFO: ::c_int = 1;
pub const SCHED_RR: ::c_int = 2;
pub const SCHED_BATCH: ::c_int = 3;
pub const SCHED_IDLE: ::c_int = 5;

// System V IPC
pub const IPC_PRIVATE: ::key_t = 0;

pub const IPC_CREAT: ::c_int = 0o1000;
pub const IPC_EXCL: ::c_int = 0o2000;
pub const IPC_NOWAIT: ::c_int = 0o4000;

pub const IPC_RMID: ::c_int = 0;
pub const IPC_SET: ::c_int = 1;
pub const IPC_STAT: ::c_int = 2;
pub const IPC_INFO: ::c_int = 3;
pub const MSG_STAT: ::c_int = 11;
pub const MSG_INFO: ::c_int = 12;

pub const MSG_NOERROR: ::c_int = 0o10000;
pub const MSG_EXCEPT: ::c_int = 0o20000;
pub const MSG_COPY: ::c_int = 0o40000;

pub const SHM_R: ::c_int = 0o400;
pub const SHM_W: ::c_int = 0o200;

pub const SHM_RDONLY: ::c_int = 0o10000;
pub const SHM_RND: ::c_int = 0o20000;
pub const SHM_REMAP: ::c_int = 0o40000;
pub const SHM_EXEC: ::c_int = 0o100000;

pub const SHM_LOCK: ::c_int = 11;
pub const SHM_UNLOCK: ::c_int = 12;

pub const SHM_HUGETLB: ::c_int = 0o4000;
pub const SHM_NORESERVE: ::c_int = 0o10000;

pub const EPOLLRDHUP: ::c_int = 0x2000;
pub const EPOLLONESHOT: ::c_int = 0x40000000;

pub const QFMT_VFS_OLD: ::c_int = 1;
pub const QFMT_VFS_V0: ::c_int = 2;

pub const SFD_CLOEXEC: ::c_int = 0x080000;

pub const EFD_SEMAPHORE: ::c_int = 0x1;

pub const NCCS: usize = 32;

pub const LOG_NFACILITIES: ::c_int = 24;

pub const SEM_FAILED: *mut ::sem_t = 0 as *mut sem_t;

pub const RB_AUTOBOOT: ::c_int = 0x01234567u32 as i32;
pub const RB_HALT_SYSTEM: ::c_int = 0xcdef0123u32 as i32;
pub const RB_ENABLE_CAD: ::c_int = 0x89abcdefu32 as i32;
pub const RB_DISABLE_CAD: ::c_int = 0x00000000u32 as i32;
pub const RB_POWER_OFF: ::c_int = 0x4321fedcu32 as i32;
pub const RB_SW_SUSPEND: ::c_int = 0xd000fce2u32 as i32;
pub const RB_KEXEC: ::c_int = 0x45584543u32 as i32;

pub const SYNC_FILE_RANGE_WAIT_BEFORE: ::c_uint = 1;
pub const SYNC_FILE_RANGE_WRITE: ::c_uint = 2;
pub const SYNC_FILE_RANGE_WAIT_AFTER: ::c_uint = 4;

pub const EAI_SYSTEM: ::c_int = -11;

f! {
    pub fn CPU_ZERO(cpuset: &mut cpu_set_t) -> () {
        for slot in cpuset.bits.iter_mut() {
            *slot = 0;
        }
    }

    pub fn CPU_SET(cpu: usize, cpuset: &mut cpu_set_t) -> () {
        let size_in_bits = 8 * mem::size_of_val(&cpuset.bits[0]); // 32, 64 etc
        let (idx, offset) = (cpu / size_in_bits, cpu % size_in_bits);
        cpuset.bits[idx] |= 1 << offset;
        ()
    }

    pub fn CPU_CLR(cpu: usize, cpuset: &mut cpu_set_t) -> () {
        let size_in_bits = 8 * mem::size_of_val(&cpuset.bits[0]); // 32, 64 etc
        let (idx, offset) = (cpu / size_in_bits, cpu % size_in_bits);
        cpuset.bits[idx] &= !(1 << offset);
        ()
    }

    pub fn CPU_ISSET(cpu: usize, cpuset: &cpu_set_t) -> bool {
        let size_in_bits = 8 * mem::size_of_val(&cpuset.bits[0]);
        let (idx, offset) = (cpu / size_in_bits, cpu % size_in_bits);
        0 != (cpuset.bits[idx] & (1 << offset))
    }

    pub fn CPU_EQUAL(set1: &cpu_set_t, set2: &cpu_set_t) -> bool {
        set1.bits == set2.bits
    }
}

extern {
    pub fn lutimes(file: *const ::c_char, times: *const ::timeval) -> ::c_int;

    pub fn setpwent();
    pub fn getpwent() -> *mut passwd;
    pub fn setspent();
    pub fn endspent();
    pub fn getspent() -> *mut spwd;
    pub fn getspnam(__name: *const ::c_char) -> *mut spwd;

    pub fn shm_open(name: *const c_char, oflag: ::c_int,
                    mode: mode_t) -> ::c_int;

    // System V IPC
    pub fn shmget(key: ::key_t, size: ::size_t, shmflg: ::c_int) -> ::c_int;
    pub fn shmat(shmid: ::c_int,
                 shmaddr: *const ::c_void,
                 shmflg: ::c_int) -> *mut ::c_void;
    pub fn shmdt(shmaddr: *const ::c_void) -> ::c_int;
    pub fn shmctl(shmid: ::c_int,
                  cmd: ::c_int,
                  buf: *mut ::shmid_ds) -> ::c_int;
    pub fn ftok(pathname: *const ::c_char, proj_id: ::c_int) -> ::key_t;
    pub fn msgctl(msqid: ::c_int, cmd: ::c_int, buf: *mut msqid_ds) -> ::c_int;
    pub fn msgget(key: ::key_t, msgflg: ::c_int) -> ::c_int;
    pub fn msgrcv(msqid: ::c_int, msgp: *mut ::c_void, msgsz: ::size_t,
                  msgtyp: ::c_long, msgflg: ::c_int) -> ::ssize_t;
    pub fn msgsnd(msqid: ::c_int, msgp: *const ::c_void, msgsz: ::size_t,
                  msgflg: ::c_int) -> ::c_int;

    pub fn mprotect(addr: *mut ::c_void, len: ::size_t, prot: ::c_int)
                    -> ::c_int;
    pub fn __errno_location() -> *mut ::c_int;

    pub fn fopen64(filename: *const c_char,
                   mode: *const c_char) -> *mut ::FILE;
    pub fn freopen64(filename: *const c_char, mode: *const c_char,
                     file: *mut ::FILE) -> *mut ::FILE;
    pub fn tmpfile64() -> *mut ::FILE;
    pub fn fgetpos64(stream: *mut ::FILE, ptr: *mut fpos64_t) -> ::c_int;
    pub fn fsetpos64(stream: *mut ::FILE, ptr: *const fpos64_t) -> ::c_int;
    pub fn fseeko64(stream: *mut ::FILE,
                    offset: ::off64_t,
                    whence: ::c_int) -> ::c_int;
    pub fn ftello64(stream: *mut ::FILE) -> ::off64_t;
    pub fn fallocate(fd: ::c_int, mode: ::c_int,
                     offset: ::off_t, len: ::off_t) -> ::c_int;
    pub fn posix_fallocate(fd: ::c_int, offset: ::off_t,
                           len: ::off_t) -> ::c_int;
    pub fn readahead(fd: ::c_int, offset: ::off64_t,
                     count: ::size_t) -> ::ssize_t;
    pub fn getxattr(path: *const c_char, name: *const c_char,
                    value: *mut ::c_void, size: ::size_t) -> ::ssize_t;
    pub fn lgetxattr(path: *const c_char, name: *const c_char,
                     value: *mut ::c_void, size: ::size_t) -> ::ssize_t;
    pub fn fgetxattr(filedes: ::c_int, name: *const c_char,
                     value: *mut ::c_void, size: ::size_t) -> ::ssize_t;
    pub fn setxattr(path: *const c_char, name: *const c_char,
                    value: *const ::c_void, size: ::size_t,
                    flags: ::c_int) -> ::c_int;
    pub fn lsetxattr(path: *const c_char, name: *const c_char,
                     value: *const ::c_void, size: ::size_t,
                     flags: ::c_int) -> ::c_int;
    pub fn fsetxattr(filedes: ::c_int, name: *const c_char,
                     value: *const ::c_void, size: ::size_t,
                     flags: ::c_int) -> ::c_int;
    pub fn listxattr(path: *const c_char, list: *mut c_char,
                     size: ::size_t) -> ::ssize_t;
    pub fn llistxattr(path: *const c_char, list: *mut c_char,
                      size: ::size_t) -> ::ssize_t;
    pub fn flistxattr(filedes: ::c_int, list: *mut c_char,
                      size: ::size_t) -> ::ssize_t;
    pub fn removexattr(path: *const c_char, name: *const c_char) -> ::c_int;
    pub fn lremovexattr(path: *const c_char, name: *const c_char) -> ::c_int;
    pub fn fremovexattr(filedes: ::c_int, name: *const c_char) -> ::c_int;
    pub fn signalfd(fd: ::c_int,
                    mask: *const ::sigset_t,
                    flags: ::c_int) -> ::c_int;
    pub fn pwritev(fd: ::c_int,
                   iov: *const ::iovec,
                   iovcnt: ::c_int,
                   offset: ::off_t) -> ::ssize_t;
    pub fn preadv(fd: ::c_int,
                  iov: *const ::iovec,
                  iovcnt: ::c_int,
                  offset: ::off_t) -> ::ssize_t;
    pub fn quotactl(cmd: ::c_int,
                    special: *const ::c_char,
                    id: ::c_int,
                    data: *mut ::c_char) -> ::c_int;
    pub fn mq_open(name: *const ::c_char, oflag: ::c_int, ...) -> ::mqd_t;
    pub fn mq_close(mqd: ::mqd_t) -> ::c_int;
    pub fn mq_unlink(name: *const ::c_char) -> ::c_int;
    pub fn mq_receive(mqd: ::mqd_t,
                      msg_ptr: *mut ::c_char,
                      msg_len: ::size_t,
                      msq_prio: *mut ::c_uint) -> ::ssize_t;
    pub fn mq_send(mqd: ::mqd_t,
                   msg_ptr: *const ::c_char,
                   msg_len: ::size_t,
                   msq_prio: ::c_uint) -> ::c_int;
    pub fn mq_getattr(mqd: ::mqd_t, attr: *mut ::mq_attr) -> ::c_int;
    pub fn mq_setattr(mqd: ::mqd_t,
                      newattr: *const ::mq_attr,
                      oldattr: *mut ::mq_attr) -> ::c_int;
    pub fn epoll_pwait(epfd: ::c_int,
                       events: *mut ::epoll_event,
                       maxevents: ::c_int,
                       timeout: ::c_int,
                       sigmask: *const ::sigset_t) -> ::c_int;
    pub fn dup3(oldfd: ::c_int, newfd: ::c_int, flags: ::c_int) -> ::c_int;
    pub fn sethostname(name: *const ::c_char, len: ::size_t) -> ::c_int;
    pub fn mkostemp(template: *mut ::c_char, flags: ::c_int) -> ::c_int;
    pub fn mkostemps(template: *mut ::c_char,
                     suffixlen: ::c_int,
                     flags: ::c_int) -> ::c_int;
    pub fn sigtimedwait(set: *const sigset_t,
                        info: *mut siginfo_t,
                        timeout: *const ::timespec) -> ::c_int;
    pub fn sigwaitinfo(set: *const sigset_t,
                       info: *mut siginfo_t) -> ::c_int;
    pub fn openpty(amaster: *mut ::c_int,
                   aslave: *mut ::c_int,
                   name: *mut ::c_char,
                   termp: *const termios,
                   winp: *const ::winsize) -> ::c_int;
    pub fn forkpty(amaster: *mut ::c_int,
                   name: *mut ::c_char,
                   termp: *const termios,
                   winp: *const ::winsize) -> ::pid_t;
    pub fn nl_langinfo_l(item: ::nl_item, locale: ::locale_t) -> *mut ::c_char;
    pub fn getnameinfo(sa: *const ::sockaddr,
                       salen: ::socklen_t,
                       host: *mut ::c_char,
                       hostlen: ::socklen_t,
                       serv: *mut ::c_char,
                       sevlen: ::socklen_t,
                       flags: ::c_int) -> ::c_int;
    pub fn prlimit(pid: ::pid_t, resource: ::c_int, new_limit: *const ::rlimit,
                   old_limit: *mut ::rlimit) -> ::c_int;
    pub fn prlimit64(pid: ::pid_t,
                     resource: ::c_int,
                     new_limit: *const ::rlimit64,
                     old_limit: *mut ::rlimit64) -> ::c_int;
    pub fn getloadavg(loadavg: *mut ::c_double, nelem: ::c_int) -> ::c_int;
    pub fn process_vm_readv(pid: ::pid_t,
                            local_iov: *const ::iovec,
                            liovcnt: ::c_ulong,
                            remote_iov: *const ::iovec,
                            riovcnt: ::c_ulong,
                            flags: ::c_ulong) -> isize;
    pub fn process_vm_writev(pid: ::pid_t,
                             local_iov: *const ::iovec,
                             liovcnt: ::c_ulong,
                             remote_iov: *const ::iovec,
                             riovcnt: ::c_ulong,
                             flags: ::c_ulong) -> isize;
    pub fn reboot(how_to: ::c_int) -> ::c_int;

    // Not available now on Android
    pub fn mkfifoat(dirfd: ::c_int, pathname: *const ::c_char,
                    mode: ::mode_t) -> ::c_int;
    pub fn if_nameindex() -> *mut if_nameindex;
    pub fn if_freenameindex(ptr: *mut if_nameindex);
    pub fn sync_file_range(fd: ::c_int, offset: ::off64_t,
                           nbytes: ::off64_t, flags: ::c_uint) -> ::c_int;
    pub fn getifaddrs(ifap: *mut *mut ::ifaddrs) -> ::c_int;
    pub fn freeifaddrs(ifa: *mut ::ifaddrs);
}

cfg_if! {
    if #[cfg(any(target_env = "musl",
                 target_os = "emscripten"))] {
        mod musl;
        pub use self::musl::*;
    } else if #[cfg(any(target_arch = "mips", target_arch = "mipsel"))] {
        mod mips;
        pub use self::mips::*;
    } else if #[cfg(any(target_arch = "s390x"))] {
        mod s390x;
        pub use self::s390x::*;
    } else if #[cfg(any(target_arch = "mips64"))] {
        mod mips64;
        pub use self::mips64::*;
    } else {
        mod other;
        pub use self::other::*;
    }
}
