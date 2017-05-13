pub type c_long = i32;
pub type c_ulong = u32;
pub type mode_t = u16;
pub type off64_t = ::c_longlong;

s! {
    pub struct sigaction {
        pub sa_sigaction: ::sighandler_t,
        pub sa_mask: ::sigset_t,
        pub sa_flags: ::c_ulong,
        pub sa_restorer: ::dox::Option<extern fn()>,
    }

    pub struct stat {
        pub st_dev: ::c_ulonglong,
        __pad0: [::c_uchar; 4],
        __st_ino: ::ino_t,
        pub st_mode: ::c_uint,
        pub st_nlink: ::c_uint,
        pub st_uid: ::uid_t,
        pub st_gid: ::gid_t,
        pub st_rdev: ::c_ulonglong,
        __pad3: [::c_uchar; 4],
        pub st_size: ::c_longlong,
        pub st_blksize: ::blksize_t,
        pub st_blocks: ::c_ulonglong,
        pub st_atime: ::c_ulong,
        pub st_atime_nsec: ::c_ulong,
        pub st_mtime: ::c_ulong,
        pub st_mtime_nsec: ::c_ulong,
        pub st_ctime: ::c_ulong,
        pub st_ctime_nsec: ::c_ulong,
        pub st_ino: ::c_ulonglong,
    }

    pub struct stat64 {
        pub st_dev: ::c_ulonglong,
        __pad0: [::c_uchar; 4],
        __st_ino: ::ino_t,
        pub st_mode: ::c_uint,
        pub st_nlink: ::c_uint,
        pub st_uid: ::uid_t,
        pub st_gid: ::gid_t,
        pub st_rdev: ::c_ulonglong,
        __pad3: [::c_uchar; 4],
        pub st_size: ::c_longlong,
        pub st_blksize: ::blksize_t,
        pub st_blocks: ::c_ulonglong,
        pub st_atime: ::c_ulong,
        pub st_atime_nsec: ::c_ulong,
        pub st_mtime: ::c_ulong,
        pub st_mtime_nsec: ::c_ulong,
        pub st_ctime: ::c_ulong,
        pub st_ctime_nsec: ::c_ulong,
        pub st_ino: ::c_ulonglong,
    }

    pub struct pthread_attr_t {
        pub flags: ::uint32_t,
        pub stack_base: *mut ::c_void,
        pub stack_size: ::size_t,
        pub guard_size: ::size_t,
        pub sched_policy: ::int32_t,
        pub sched_priority: ::int32_t,
    }

    pub struct pthread_mutex_t { value: ::c_int }

    pub struct pthread_cond_t { value: ::c_int }

    pub struct pthread_rwlock_t {
        lock: pthread_mutex_t,
        cond: pthread_cond_t,
        numLocks: ::c_int,
        writerThreadId: ::c_int,
        pendingReaders: ::c_int,
        pendingWriters: ::c_int,
        attr: i32,
        __reserved: [::c_char; 12],
    }

    pub struct passwd {
        pub pw_name: *mut ::c_char,
        pub pw_passwd: *mut ::c_char,
        pub pw_uid: ::uid_t,
        pub pw_gid: ::gid_t,
        pub pw_dir: *mut ::c_char,
        pub pw_shell: *mut ::c_char,
    }

    pub struct statfs {
        pub f_type: ::uint32_t,
        pub f_bsize: ::uint32_t,
        pub f_blocks: ::uint64_t,
        pub f_bfree: ::uint64_t,
        pub f_bavail: ::uint64_t,
        pub f_files: ::uint64_t,
        pub f_ffree: ::uint64_t,
        pub f_fsid: ::__fsid_t,
        pub f_namelen: ::uint32_t,
        pub f_frsize: ::uint32_t,
        pub f_flags: ::uint32_t,
        pub f_spare: [::uint32_t; 4],
    }

    pub struct sysinfo {
        pub uptime: ::c_long,
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
        pub _f: [::c_char; 8],
    }
}

pub const SYS_gettid: ::c_long = 224;
pub const PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
    value: 0,
};
pub const PTHREAD_COND_INITIALIZER: pthread_cond_t = pthread_cond_t {
    value: 0,
};
pub const PTHREAD_RWLOCK_INITIALIZER: pthread_rwlock_t = pthread_rwlock_t {
    lock: PTHREAD_MUTEX_INITIALIZER,
    cond: PTHREAD_COND_INITIALIZER,
    numLocks: 0,
    writerThreadId: 0,
    pendingReaders: 0,
    pendingWriters: 0,
    attr: 0,
    __reserved: [0; 12],
};
pub const PTHREAD_STACK_MIN: ::size_t = 4096 * 2;
pub const CPU_SETSIZE: ::size_t = 32;
pub const __CPU_BITS: ::size_t = 32;

pub const UT_LINESIZE: usize = 8;
pub const UT_NAMESIZE: usize = 8;
pub const UT_HOSTSIZE: usize = 16;

extern {
    pub fn timegm64(tm: *const ::tm) -> ::time64_t;
}
