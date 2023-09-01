//! Platform-specific parts of the standard library
//! that can be plugged-in at runtime.
//!
//! Using these modules, code can set an implementation for
//! each platform-specific part of the standard library at
//! runtime. It does so via the `set_impl` functions.
//!
//! This is primarily geared toward experimental platforms such
//! as new kernels and bare-bones environments, where you might
//! want to use the standard library without recompiling
//! everything or adding support upstream.
//!
//! # Initial state
//!
//! Initially, as no implementation has been defined, most
//! of these parts panic with this message:
//!
//! ```text
//! std::os::thread::IMPL has not been initialized at this point.
//! ```
//!
//! There are three exceptions:
//!
//! - There is a default allocator, allowing for 16KiB of heap use.
//!   It implements the sequential fit algorithm.
//! - Initially, locking primitives are functional and behave as spinlocks.
//! - Before `os::set_impl` has been called, any call to `abort_internal`
//!   will result in an infinite loop.
//!
//! These should be set/changed as soon as possible, as they are used internally.
//!
//! # To do
//!
//! - thread parking support
//! - thread-safe Once support
//! - invocation arguments (returns an empty iterator at the moment)
//! - `pipe::read2`
//!

#![unstable(issue = "none", feature = "std_internals")]

#[doc(hidden)]
#[macro_export]
macro_rules! custom_os_impl {
    ($module:ident, $method:ident $(, $arg:expr)*) => {{
        let errmsg = concat!(
            "std::os::", stringify!($module), "::IMPL",
            " has not been initialized at this point",
        );

        let rwlock = &crate::os::custom::$module::IMPL;
        let reader = rwlock.read().expect("poisoned lock");
        let some_impl = reader.as_ref().expect(errmsg);

        some_impl.$method($($arg,)*)
    }};
}

macro_rules! static_rwlock_box_impl {
    ($api:ident) => {
        pub(crate) static IMPL: RwLock<Option<Box<dyn $api>>> = RwLock::new(None);

        /// Sets the implementation singleton
        ///
        /// This parameter takes a `transition` closure responsible
        /// for properly transitioning from the previous implementation
        /// to a new one.
        ///
        /// Initially, there is no implementation; the first time this is called,
        /// the closure parameter will be `None`.
        ///
        /// Removing an implementation (i.e. setting the internal singleton to `None`)
        /// is intentionally not allowed.
        pub fn set_impl<F: FnMut(Option<Box<dyn $api>>) -> Box<dyn $api>>(mut transition: F) {
            let mut writer = IMPL.write().expect("poisoned lock");
            let maybe_impl = core::mem::replace(&mut *writer, None);
            let new_impl = transition(maybe_impl);
            *writer = Some(new_impl);
        }
    };
}

/// Platform-specific allocator
pub mod alloc {
    use crate::alloc::GlobalAlloc;
    use crate::sync::RwLock;

    static_rwlock_box_impl!(Allocator);

    /// Platform-specific allocator
    pub trait Allocator: GlobalAlloc + Send + Sync {}
}

/// Platform-specific interface to a filesystem
pub mod fs {
    use crate::io;
    use crate::path::{Path, PathBuf};
    use crate::sync::RwLock;

    #[doc(inline)]
    pub use crate::sys::fs::{
        DirEntry, File, FileApi, FileAttr, FilePermissions, FileTimes, FileType, OpenOptions,
        ReadDir, ReadDirApi,
    };

    static_rwlock_box_impl!(FilesystemInterface);

    /// Platform-specific interface to a filesystem
    pub trait FilesystemInterface: Send + Sync {
        fn open(&self, path: &Path, opts: &OpenOptions) -> io::Result<File>;
        fn mkdir(&self, p: &Path) -> io::Result<()>;
        fn read_dir(&self, p: &Path) -> io::Result<ReadDir>;
        fn unlink(&self, p: &Path) -> io::Result<()>;
        fn rename(&self, old: &Path, new: &Path) -> io::Result<()>;
        fn set_perm(&self, p: &Path, perm: FilePermissions) -> io::Result<()>;
        fn rmdir(&self, p: &Path) -> io::Result<()>;
        fn remove_dir_all(&self, path: &Path) -> io::Result<()>;
        fn try_exists(&self, path: &Path) -> io::Result<bool>;
        fn readlink(&self, p: &Path) -> io::Result<PathBuf>;
        fn symlink(&self, original: &Path, link: &Path) -> io::Result<()>;
        fn link(&self, src: &Path, dst: &Path) -> io::Result<()>;
        fn stat(&self, p: &Path) -> io::Result<FileAttr>;
        fn lstat(&self, p: &Path) -> io::Result<FileAttr>;
        fn canonicalize(&self, p: &Path) -> io::Result<PathBuf>;
        fn copy(&self, from: &Path, to: &Path) -> io::Result<u64>;
    }
}

/// Platform-specific management of `AtomicU32`-based futexes
pub mod futex {
    use crate::sync::{atomic::AtomicU32, RwLock};
    use crate::time::Duration;

    // by default (None) => spinlock
    static_rwlock_box_impl!(FutexManager);

    /// Platform-specific management of `AtomicU32`-based futexes
    pub trait FutexManager: Send + Sync {
        /// Wait for a futex_wake operation to wake us.
        ///
        /// Returns directly if the futex doesn't hold the expected value.
        ///
        /// Returns false on timeout, and true in all other cases.
        fn futex_wait(&self, futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool;

        /// Wake up one thread that's blocked on futex_wait on this futex.
        ///
        /// Returns true if this actually woke up such a thread,
        /// or false if no thread was waiting on this futex.
        ///
        /// On some platforms, this always returns false.
        fn futex_wake(&self, futex: &AtomicU32) -> bool;

        /// Wake up all threads that are waiting on futex_wait on this futex.
        fn futex_wake_all(&self, futex: &AtomicU32);
    }
}

/// Platform-specific interface to a network
pub mod net {
    use crate::io;
    use crate::net::SocketAddr;
    use crate::sync::RwLock;
    use crate::time::Duration;

    #[doc(inline)]
    pub use crate::sys::net::{
        LookupHost, TcpListener, TcpListenerApi, TcpStream, TcpStreamApi, UdpSocket, UdpSocketApi,
    };

    static_rwlock_box_impl!(NetworkInterface);

    /// Platform-specific interface to a network
    pub trait NetworkInterface: Send + Sync {
        fn tcp_connect(
            &self,
            addr: &SocketAddr,
            timeout: Option<Duration>,
        ) -> io::Result<TcpStream>;
        fn tcp_bind(&self, addr: &SocketAddr) -> io::Result<TcpListener>;
        fn udp_bind(&self, addr: &SocketAddr) -> io::Result<UdpSocket>;
        fn lookup_str(&self, v: &str) -> io::Result<LookupHost>;
        fn lookup_tuple(&self, v: (&str, u16)) -> io::Result<LookupHost>;
    }
}

/// Platform-specific interface to the running operating system
pub mod os {
    use crate::ffi::{OsStr, OsString};
    use crate::io;
    use crate::path::{Path, PathBuf};
    use crate::sync::RwLock;

    #[doc(inline)]
    pub use crate::sys::os::{Env, JoinPathsError, SplitPaths, Variable};

    static_rwlock_box_impl!(Os);

    /// Platform-specific interface to the running operating system
    pub trait Os: Send + Sync {
        fn errno(&self) -> i32;
        fn error_string(&self, errno: i32) -> String;

        fn current_exe(&self) -> io::Result<PathBuf>;
        fn env(&self) -> Env;
        fn get_env(&self, variable: &OsStr) -> Option<OsString>;
        fn set_env(&self, variable: &OsStr, value: &OsStr) -> io::Result<()>;
        fn unset_env(&self, variable: &OsStr) -> io::Result<()>;
        fn env_path_delim(&self) -> &'static str;

        fn getcwd(&self) -> io::Result<PathBuf>;
        fn temp_dir(&self) -> PathBuf;
        fn home_dir(&self) -> Option<PathBuf>;
        fn chdir(&self, path: &Path) -> io::Result<()>;

        fn exit(&self, code: i32) -> !;
        fn get_pid(&self) -> u32;

        fn decode_error_kind(&self, errno: i32) -> crate::io::ErrorKind;
        fn hashmap_random_keys(&self) -> (u64, u64);
    }
}

/// Platform-specific management of processes
pub mod process {
    use crate::io;
    use crate::sync::RwLock;

    #[doc(inline)]
    pub use crate::sys_common::process::{CommandEnv, CommandEnvs};

    #[doc(inline)]
    pub use crate::sys::process::{Command, ExitStatus, Process, ProcessApi, Stdio, StdioPipes};

    static_rwlock_box_impl!(ProcessManager);

    /// Platform-specific management of processes
    pub trait ProcessManager: Send + Sync {
        fn spawn(&self, command: &Command) -> io::Result<(Process, StdioPipes)>;
    }
}

/// Platform-specific standard IO interface
pub mod stdio {
    use crate::io;
    use crate::sync::RwLock;

    static_rwlock_box_impl!(StdioInterface);

    /// Platform-specific standard IO interface
    pub trait StdioInterface: Send + Sync {
        fn read_stdin(&self, buf: &mut [u8]) -> io::Result<usize>;
        fn write_stdout(&self, buf: &[u8]) -> io::Result<usize>;
        fn flush_stdout(&self) -> io::Result<()>;
        fn write_stderr(&self, buf: &[u8]) -> io::Result<usize>;
        fn flush_stderr(&self) -> io::Result<()>;
        fn is_ebadf(&self, err: &io::Error) -> bool;
        fn panic_output(&self) -> Option<Vec<u8>>;
    }
}

/// Platform-specific management of threads
pub mod thread {
    use crate::ffi::CStr;
    use crate::io;
    use crate::num::NonZeroUsize;
    use crate::sync::RwLock;
    use crate::time::Duration;

    #[doc(inline)]
    pub use crate::sys::thread::{Thread, ThreadApi};

    static_rwlock_box_impl!(ThreadManager);

    /// Platform-specific management of threads
    pub trait ThreadManager: Send + Sync {
        /// unsafe: see thread::Builder::spawn_unchecked for safety requirements
        unsafe fn new(&self, stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread>;
        fn yield_now(&self);
        fn set_name(&self, name: &CStr);
        fn sleep(&self, dur: Duration);
        fn join(&self, thread: &Thread);
        fn available_parallelism(&self) -> io::Result<NonZeroUsize>;

        // todo: thread parking
    }
}

/// Platform-specific timer interface
pub mod time {
    use crate::sync::RwLock;

    pub use crate::sys::time::{Instant, SystemTime};

    static_rwlock_box_impl!(Timer);

    /// Platform-specific timer interface
    pub trait Timer: Send + Sync {
        fn now_instant(&self) -> Instant;
        fn now_systime(&self) -> SystemTime;
    }
}
