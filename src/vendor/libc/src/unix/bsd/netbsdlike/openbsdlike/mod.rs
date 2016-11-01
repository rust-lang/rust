pub type c_long = i64;
pub type c_ulong = u64;
pub type clock_t = i64;
pub type suseconds_t = i64;
pub type dev_t = i32;
pub type sigset_t = ::c_uint;
pub type blksize_t = ::int32_t;
pub type fsblkcnt_t = ::uint64_t;
pub type fsfilcnt_t = ::uint64_t;
pub type pthread_attr_t = *mut ::c_void;
pub type pthread_mutex_t = *mut ::c_void;
pub type pthread_mutexattr_t = *mut ::c_void;
pub type pthread_cond_t = *mut ::c_void;
pub type pthread_condattr_t = *mut ::c_void;
pub type pthread_rwlock_t = *mut ::c_void;

s! {
    pub struct dirent {
        pub d_fileno: ::ino_t,
        pub d_off: ::off_t,
        pub d_reclen: u16,
        pub d_type: u8,
        pub d_namlen: u8,
        __d_padding: [u8; 4],
        pub d_name: [::c_char; 256],
    }

    pub struct glob_t {
        pub gl_pathc:   ::c_int,
        pub gl_matchc:  ::c_int,
        pub gl_offs:    ::c_int,
        pub gl_flags:   ::c_int,
        pub gl_pathv:   *mut *mut ::c_char,
        __unused1: *mut ::c_void,
        __unused2: *mut ::c_void,
        __unused3: *mut ::c_void,
        __unused4: *mut ::c_void,
        __unused5: *mut ::c_void,
        __unused6: *mut ::c_void,
        __unused7: *mut ::c_void,
    }

    pub struct kevent {
        pub ident: ::uintptr_t,
        pub filter: ::c_short,
        pub flags: ::c_ushort,
        pub fflags: ::c_uint,
        pub data: ::int64_t,
        pub udata: *mut ::c_void,
    }

    pub struct stat {
        pub st_mode: ::mode_t,
        pub st_dev: ::dev_t,
        pub st_ino: ::ino_t,
        pub st_nlink: ::nlink_t,
        pub st_uid: ::uid_t,
        pub st_gid: ::gid_t,
        pub st_rdev: ::dev_t,
        pub st_atime: ::time_t,
        pub st_atime_nsec: ::c_long,
        pub st_mtime: ::time_t,
        pub st_mtime_nsec: ::c_long,
        pub st_ctime: ::time_t,
        pub st_ctime_nsec: ::c_long,
        pub st_size: ::off_t,
        pub st_blocks: ::blkcnt_t,
        pub st_blksize: ::blksize_t,
        pub st_flags: ::uint32_t,
        pub st_gen: ::uint32_t,
        pub st_birthtime: ::time_t,
        pub st_birthtime_nsec: ::c_long,
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

    pub struct addrinfo {
        pub ai_flags: ::c_int,
        pub ai_family: ::c_int,
        pub ai_socktype: ::c_int,
        pub ai_protocol: ::c_int,
        pub ai_addrlen: ::socklen_t,
        pub ai_addr: *mut ::sockaddr,
        pub ai_canonname: *mut ::c_char,
        pub ai_next: *mut ::addrinfo,
    }

    pub struct sockaddr_storage {
        pub ss_len: u8,
        pub ss_family: ::sa_family_t,
        __ss_pad1: [u8; 6],
        __ss_pad2: i64,
        __ss_pad3: [u8; 240],
    }

    pub struct siginfo_t {
        pub si_signo: ::c_int,
        pub si_code: ::c_int,
        pub si_errno: ::c_int,
        pub si_addr: *mut ::c_char,
        __pad: [u8; 108],
    }

    pub struct Dl_info {
        pub dli_fname: *const ::c_char,
        pub dli_fbase: *mut ::c_void,
        pub dli_sname: *const ::c_char,
        pub dli_saddr: *mut ::c_void,
    }

    pub struct lastlog {
        ll_time: ::time_t,
        ll_line: [::c_char; UT_LINESIZE],
        ll_host: [::c_char; UT_HOSTSIZE],
    }

    pub struct utmp {
        pub ut_line: [::c_char; UT_LINESIZE],
        pub ut_name: [::c_char; UT_NAMESIZE],
        pub ut_host: [::c_char; UT_HOSTSIZE],
        pub ut_time: ::time_t,
    }
}

pub const UT_NAMESIZE: usize = 32;
pub const UT_LINESIZE: usize = 8;
pub const UT_HOSTSIZE: usize = 256;

pub const O_CLOEXEC: ::c_int = 0x10000;

pub const MS_SYNC : ::c_int = 0x0002;
pub const MS_INVALIDATE : ::c_int = 0x0004;

pub const PTHREAD_STACK_MIN : ::size_t = 2048;

pub const ENOATTR : ::c_int = 83;
pub const EILSEQ : ::c_int = 84;
pub const EOVERFLOW : ::c_int = 87;
pub const ECANCELED : ::c_int = 88;
pub const EIDRM : ::c_int = 89;
pub const ENOMSG : ::c_int = 90;
pub const ENOTSUP : ::c_int = 91;
pub const ELAST : ::c_int = 91;

pub const F_DUPFD_CLOEXEC : ::c_int = 10;

pub const RLIM_NLIMITS: ::c_int = 9;

pub const SO_SNDTIMEO: ::c_int = 0x1005;
pub const SO_RCVTIMEO: ::c_int = 0x1006;

pub const IPV6_JOIN_GROUP: ::c_int = 12;
pub const IPV6_LEAVE_GROUP: ::c_int = 13;

pub const O_DSYNC : ::c_int = 128;

pub const MAP_RENAME : ::c_int = 0x0000;
pub const MAP_NORESERVE : ::c_int = 0x0000;
pub const MAP_HASSEMAPHORE : ::c_int = 0x0000;

pub const EIPSEC : ::c_int = 82;
pub const ENOMEDIUM : ::c_int = 85;
pub const EMEDIUMTYPE : ::c_int = 86;

pub const EAI_SYSTEM: ::c_int = -11;

pub const RUSAGE_THREAD: ::c_int = 1;

pub const MAP_COPY : ::c_int = 0x0002;
pub const MAP_NOEXTEND : ::c_int = 0x0000;

pub const _SC_CLK_TCK : ::c_int = 3;
pub const _SC_IOV_MAX : ::c_int = 51;
pub const _SC_GETGR_R_SIZE_MAX : ::c_int = 100;
pub const _SC_GETPW_R_SIZE_MAX : ::c_int = 101;
pub const _SC_LOGIN_NAME_MAX : ::c_int = 102;
pub const _SC_MQ_PRIO_MAX : ::c_int = 59;
pub const _SC_NPROCESSORS_ONLN : ::c_int = 503;
pub const _SC_THREADS : ::c_int = 91;
pub const _SC_THREAD_ATTR_STACKADDR : ::c_int = 77;
pub const _SC_THREAD_ATTR_STACKSIZE : ::c_int = 78;
pub const _SC_THREAD_DESTRUCTOR_ITERATIONS : ::c_int = 80;
pub const _SC_THREAD_KEYS_MAX : ::c_int = 81;
pub const _SC_THREAD_PRIO_INHERIT : ::c_int = 82;
pub const _SC_THREAD_PRIO_PROTECT : ::c_int = 83;
pub const _SC_THREAD_PRIORITY_SCHEDULING : ::c_int = 84;
pub const _SC_THREAD_PROCESS_SHARED : ::c_int = 85;
pub const _SC_THREAD_SAFE_FUNCTIONS : ::c_int = 103;
pub const _SC_THREAD_STACK_MIN : ::c_int = 89;
pub const _SC_THREAD_THREADS_MAX : ::c_int = 90;
pub const _SC_TTY_NAME_MAX : ::c_int = 107;
pub const _SC_ATEXIT_MAX : ::c_int = 46;
pub const _SC_AIO_LISTIO_MAX : ::c_int = 42;
pub const _SC_AIO_MAX : ::c_int = 43;
pub const _SC_ASYNCHRONOUS_IO : ::c_int = 45;
pub const _SC_MAPPED_FILES : ::c_int = 53;
pub const _SC_MEMLOCK : ::c_int = 54;
pub const _SC_MEMLOCK_RANGE : ::c_int = 55;
pub const _SC_MEMORY_PROTECTION : ::c_int = 56;
pub const _SC_MESSAGE_PASSING : ::c_int = 57;
pub const _SC_MQ_OPEN_MAX : ::c_int = 58;
pub const _SC_PRIORITY_SCHEDULING : ::c_int = 61;
pub const _SC_SEMAPHORES : ::c_int = 67;
pub const _SC_SHARED_MEMORY_OBJECTS : ::c_int = 68;
pub const _SC_SYNCHRONIZED_IO : ::c_int = 75;
pub const _SC_TIMERS : ::c_int = 94;
pub const _SC_XOPEN_CRYPT : ::c_int = 117;
pub const _SC_XOPEN_ENH_I18N : ::c_int = 118;
pub const _SC_XOPEN_LEGACY : ::c_int = 119;
pub const _SC_XOPEN_REALTIME : ::c_int = 120;
pub const _SC_XOPEN_REALTIME_THREADS : ::c_int = 121;
pub const _SC_XOPEN_UNIX : ::c_int = 123;
pub const _SC_XOPEN_VERSION : ::c_int = 125;
pub const _SC_SEM_NSEMS_MAX : ::c_int = 31;
pub const _SC_SEM_VALUE_MAX : ::c_int = 32;
pub const _SC_AIO_PRIO_DELTA_MAX : ::c_int = 44;
pub const _SC_DELAYTIMER_MAX : ::c_int = 50;
pub const _SC_PRIORITIZED_IO : ::c_int = 60;
pub const _SC_REALTIME_SIGNALS : ::c_int = 64;
pub const _SC_RTSIG_MAX : ::c_int = 66;
pub const _SC_SIGQUEUE_MAX : ::c_int = 70;
pub const _SC_TIMER_MAX : ::c_int = 93;
pub const _SC_HOST_NAME_MAX : ::c_int = 33;

pub const FD_SETSIZE: usize = 1024;

pub const ST_NOSUID: ::c_ulong = 2;

pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = 0 as *mut _;
pub const PTHREAD_COND_INITIALIZER: pthread_cond_t = 0 as *mut _;
pub const PTHREAD_RWLOCK_INITIALIZER: pthread_rwlock_t = 0 as *mut _;

pub const PTHREAD_MUTEX_ERRORCHECK: ::c_int = 1;
pub const PTHREAD_MUTEX_RECURSIVE: ::c_int = 2;
pub const PTHREAD_MUTEX_NORMAL: ::c_int = 3;
pub const PTHREAD_MUTEX_STRICT_NP: ::c_int = 4;
pub const PTHREAD_MUTEX_DEFAULT: ::c_int = PTHREAD_MUTEX_STRICT_NP;

pub const EVFILT_AIO: ::int16_t = -3;
pub const EVFILT_PROC: ::int16_t = -5;
pub const EVFILT_READ: ::int16_t = -1;
pub const EVFILT_SIGNAL: ::int16_t = -6;
pub const EVFILT_TIMER: ::int16_t = -7;
pub const EVFILT_VNODE: ::int16_t = -4;
pub const EVFILT_WRITE: ::int16_t = -2;

pub const EV_ADD: ::uint16_t = 0x1;
pub const EV_DELETE: ::uint16_t = 0x2;
pub const EV_ENABLE: ::uint16_t = 0x4;
pub const EV_DISABLE: ::uint16_t = 0x8;
pub const EV_ONESHOT: ::uint16_t = 0x10;
pub const EV_CLEAR: ::uint16_t = 0x20;
pub const EV_FLAG1: ::uint16_t = 0x2000;
pub const EV_ERROR: ::uint16_t = 0x4000;
pub const EV_EOF: ::uint16_t = 0x8000;
pub const EV_SYSFLAGS: ::uint16_t = 0xf000;

pub const NOTE_LOWAT: ::uint32_t = 0x00000001;
pub const NOTE_EOF: ::uint32_t = 0x00000002;
pub const NOTE_DELETE: ::uint32_t = 0x00000001;
pub const NOTE_WRITE: ::uint32_t = 0x00000002;
pub const NOTE_EXTEND: ::uint32_t = 0x00000004;
pub const NOTE_ATTRIB: ::uint32_t = 0x00000008;
pub const NOTE_LINK: ::uint32_t = 0x00000010;
pub const NOTE_RENAME: ::uint32_t = 0x00000020;
pub const NOTE_REVOKE: ::uint32_t = 0x00000040;
pub const NOTE_TRUNCATE: ::uint32_t = 0x00000080;
pub const NOTE_EXIT: ::uint32_t = 0x80000000;
pub const NOTE_FORK: ::uint32_t = 0x40000000;
pub const NOTE_EXEC: ::uint32_t = 0x20000000;
pub const NOTE_PDATAMASK: ::uint32_t = 0x000fffff;
pub const NOTE_PCTRLMASK: ::uint32_t = 0xf0000000;
pub const NOTE_TRACK: ::uint32_t = 0x00000001;
pub const NOTE_TRACKERR: ::uint32_t = 0x00000002;
pub const NOTE_CHILD: ::uint32_t = 0x00000004;

pub const TMP_MAX : ::c_uint = 0x7fffffff;

pub const NI_MAXHOST: ::size_t = 256;

pub const RTLD_LOCAL: ::c_int = 0;
pub const CTL_MAXNAME: ::c_int = 12;
pub const CTLTYPE_NODE: ::c_int = 1;
pub const CTLTYPE_INT: ::c_int = 2;
pub const CTLTYPE_STRING: ::c_int = 3;
pub const CTLTYPE_QUAD: ::c_int = 4;
pub const CTLTYPE_STRUCT: ::c_int = 5;
pub const CTL_UNSPEC: ::c_int = 0;
pub const CTL_KERN: ::c_int = 1;
pub const CTL_VM: ::c_int = 2;
pub const CTL_FS: ::c_int = 3;
pub const CTL_NET: ::c_int = 4;
pub const CTL_DEBUG: ::c_int = 5;
pub const CTL_HW: ::c_int = 6;
pub const CTL_MACHDEP: ::c_int = 7;
pub const CTL_DDB: ::c_int = 9;
pub const CTL_VFS: ::c_int = 10;
pub const CTL_MAXID: ::c_int = 11;
pub const KERN_OSTYPE: ::c_int = 1;
pub const KERN_OSRELEASE: ::c_int = 2;
pub const KERN_OSREV: ::c_int = 3;
pub const KERN_VERSION: ::c_int = 4;
pub const KERN_MAXVNODES: ::c_int = 5;
pub const KERN_MAXPROC: ::c_int = 6;
pub const KERN_MAXFILES: ::c_int = 7;
pub const KERN_ARGMAX: ::c_int = 8;
pub const KERN_SECURELVL: ::c_int = 9;
pub const KERN_HOSTNAME: ::c_int = 10;
pub const KERN_HOSTID: ::c_int = 11;
pub const KERN_CLOCKRATE: ::c_int = 12;
pub const KERN_PROF: ::c_int = 16;
pub const KERN_POSIX1: ::c_int = 17;
pub const KERN_NGROUPS: ::c_int = 18;
pub const KERN_JOB_CONTROL: ::c_int = 19;
pub const KERN_SAVED_IDS: ::c_int = 20;
pub const KERN_BOOTTIME: ::c_int = 21;
pub const KERN_DOMAINNAME: ::c_int = 22;
pub const KERN_MAXPARTITIONS: ::c_int = 23;
pub const KERN_RAWPARTITION: ::c_int = 24;
pub const KERN_MAXTHREAD: ::c_int = 25;
pub const KERN_NTHREADS: ::c_int = 26;
pub const KERN_OSVERSION: ::c_int = 27;
pub const KERN_SOMAXCONN: ::c_int = 28;
pub const KERN_SOMINCONN: ::c_int = 29;
pub const KERN_USERMOUNT: ::c_int = 30;
pub const KERN_NOSUIDCOREDUMP: ::c_int = 32;
pub const KERN_FSYNC: ::c_int = 33;
pub const KERN_SYSVMSG: ::c_int = 34;
pub const KERN_SYSVSEM: ::c_int = 35;
pub const KERN_SYSVSHM: ::c_int = 36;
pub const KERN_ARND: ::c_int = 37;
pub const KERN_MSGBUFSIZE: ::c_int = 38;
pub const KERN_MALLOCSTATS: ::c_int = 39;
pub const KERN_CPTIME: ::c_int = 40;
pub const KERN_NCHSTATS: ::c_int = 41;
pub const KERN_FORKSTAT: ::c_int = 42;
pub const KERN_NSELCOLL: ::c_int = 43;
pub const KERN_TTY: ::c_int = 44;
pub const KERN_CCPU: ::c_int = 45;
pub const KERN_FSCALE: ::c_int = 46;
pub const KERN_NPROCS: ::c_int = 47;
pub const KERN_MSGBUF: ::c_int = 48;
pub const KERN_POOL: ::c_int = 49;
pub const KERN_STACKGAPRANDOM: ::c_int = 50;
pub const KERN_SYSVIPC_INFO: ::c_int = 51;
pub const KERN_SPLASSERT: ::c_int = 54;
pub const KERN_PROC_ARGS: ::c_int = 55;
pub const KERN_NFILES: ::c_int = 56;
pub const KERN_TTYCOUNT: ::c_int = 57;
pub const KERN_NUMVNODES: ::c_int = 58;
pub const KERN_MBSTAT: ::c_int = 59;
pub const KERN_SEMINFO: ::c_int = 61;
pub const KERN_SHMINFO: ::c_int = 62;
pub const KERN_INTRCNT: ::c_int = 63;
pub const KERN_WATCHDOG: ::c_int = 64;
pub const KERN_PROC: ::c_int = 66;
pub const KERN_MAXCLUSTERS: ::c_int = 67;
pub const KERN_EVCOUNT: ::c_int = 68;
pub const KERN_TIMECOUNTER: ::c_int = 69;
pub const KERN_MAXLOCKSPERUID: ::c_int = 70;
pub const KERN_CPTIME2: ::c_int = 71;
pub const KERN_CACHEPCT: ::c_int = 72;
pub const KERN_FILE: ::c_int = 73;
pub const KERN_CONSDEV: ::c_int = 75;
pub const KERN_NETLIVELOCKS: ::c_int = 76;
pub const KERN_POOL_DEBUG: ::c_int = 77;
pub const KERN_PROC_CWD: ::c_int = 78;
pub const KERN_PROC_NOBROADCASTKILL: ::c_int = 79;
pub const KERN_PROC_VMMAP: ::c_int = 80;
pub const KERN_GLOBAL_PTRACE: ::c_int = 81;
pub const KERN_CONSBUFSIZE: ::c_int = 82;
pub const KERN_CONSBUF: ::c_int = 83;
pub const KERN_MAXID: ::c_int = 84;
pub const KERN_PROC_ALL: ::c_int = 0;
pub const KERN_PROC_PID: ::c_int = 1;
pub const KERN_PROC_PGRP: ::c_int = 2;
pub const KERN_PROC_SESSION: ::c_int = 3;
pub const KERN_PROC_TTY: ::c_int = 4;
pub const KERN_PROC_UID: ::c_int = 5;
pub const KERN_PROC_RUID: ::c_int = 6;
pub const KERN_PROC_KTHREAD: ::c_int = 7;
pub const KERN_PROC_SHOW_THREADS: ::c_int = 0x40000000;
pub const KERN_SYSVIPC_MSG_INFO: ::c_int = 1;
pub const KERN_SYSVIPC_SEM_INFO: ::c_int = 2;
pub const KERN_SYSVIPC_SHM_INFO: ::c_int = 3;
pub const KERN_PROC_ARGV: ::c_int = 1;
pub const KERN_PROC_NARGV: ::c_int = 2;
pub const KERN_PROC_ENV: ::c_int = 3;
pub const KERN_PROC_NENV: ::c_int = 4;
pub const KI_NGROUPS: ::c_int = 16;
pub const KI_MAXCOMLEN: ::c_int = 24;
pub const KI_WMESGLEN: ::c_int = 8;
pub const KI_MAXLOGNAME: ::c_int = 32;
pub const KI_EMULNAMELEN: ::c_int = 8;

extern {
    pub fn getnameinfo(sa: *const ::sockaddr,
                       salen: ::socklen_t,
                       host: *mut ::c_char,
                       hostlen: ::size_t,
                       serv: *mut ::c_char,
                       servlen: ::size_t,
                       flags: ::c_int) -> ::c_int;
    pub fn kevent(kq: ::c_int,
                  changelist: *const ::kevent,
                  nchanges: ::c_int,
                  eventlist: *mut ::kevent,
                  nevents: ::c_int,
                  timeout: *const ::timespec) -> ::c_int;
    pub fn mprotect(addr: *mut ::c_void, len: ::size_t, prot: ::c_int)
                    -> ::c_int;
    pub fn pthread_main_np() -> ::c_int;
    pub fn pthread_set_name_np(tid: ::pthread_t, name: *const ::c_char);
    pub fn pthread_stackseg_np(thread: ::pthread_t,
                               sinfo: *mut ::stack_t) -> ::c_int;
    pub fn sysctl(name: *const ::c_int,
                  namelen: ::c_uint,
                  oldp: *mut ::c_void,
                  oldlenp: *mut ::size_t,
                  newp: *mut ::c_void,
                  newlen: ::size_t)
                  -> ::c_int;
    pub fn getentropy(buf: *mut ::c_void, buflen: ::size_t) -> ::c_int;
}

cfg_if! {
    if #[cfg(target_os = "openbsd")] {
        mod openbsd;
        pub use self::openbsd::*;
    } else if #[cfg(target_os = "bitrig")] {
        mod bitrig;
        pub use self::bitrig::*;
    } else {
        // Unknown target_os
    }
}
