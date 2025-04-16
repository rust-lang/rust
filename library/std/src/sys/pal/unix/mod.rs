#![allow(missing_docs, nonstandard_style)]

use crate::io::ErrorKind;

#[cfg(not(target_os = "espidf"))]
#[macro_use]
pub mod weak;

pub mod env;
#[cfg(target_os = "fuchsia")]
pub mod fuchsia;
pub mod futex;
#[cfg(any(target_os = "linux", target_os = "android"))]
pub mod kernel_copy;
#[cfg(target_os = "linux")]
pub mod linux;
pub mod os;
pub mod pipe;
pub mod stack_overflow;
pub mod sync;
pub mod thread;
pub mod thread_parking;
pub mod time;

#[cfg(target_os = "espidf")]
pub fn init(_argc: isize, _argv: *const *const u8, _sigpipe: u8) {}

#[cfg(not(target_os = "espidf"))]
#[cfg_attr(target_os = "vita", allow(unused_variables))]
// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
// See `fn init()` in `library/std/src/rt.rs` for docs on `sigpipe`.
pub unsafe fn init(argc: isize, argv: *const *const u8, sigpipe: u8) {
    // The standard streams might be closed on application startup. To prevent
    // std::io::{stdin, stdout,stderr} objects from using other unrelated file
    // resources opened later, we reopen standards streams when they are closed.
    sanitize_standard_fds();

    // By default, some platforms will send a *signal* when an EPIPE error
    // would otherwise be delivered. This runtime doesn't install a SIGPIPE
    // handler, causing it to kill the program, which isn't exactly what we
    // want!
    //
    // Hence, we set SIGPIPE to ignore when the program starts up in order
    // to prevent this problem. Use `-Zon-broken-pipe=...` to alter this
    // behavior.
    reset_sigpipe(sigpipe);

    stack_overflow::init();
    #[cfg(not(target_os = "vita"))]
    crate::sys::args::init(argc, argv);

    // Normally, `thread::spawn` will call `Thread::set_name` but since this thread
    // already exists, we have to call it ourselves. We only do this on Apple targets
    // because some unix-like operating systems such as Linux share process-id and
    // thread-id for the main thread and so renaming the main thread will rename the
    // process and we only want to enable this on platforms we've tested.
    if cfg!(target_vendor = "apple") {
        thread::Thread::set_name(&c"main");
    }

    unsafe fn sanitize_standard_fds() {
        // fast path with a single syscall for systems with poll()
        #[cfg(not(any(
            miri,
            target_os = "emscripten",
            target_os = "fuchsia",
            target_os = "vxworks",
            target_os = "redox",
            target_os = "l4re",
            target_os = "horizon",
            target_os = "vita",
            target_os = "rtems",
            // The poll on Darwin doesn't set POLLNVAL for closed fds.
            target_vendor = "apple",
        )))]
        'poll: {
            #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
            use libc::open as open64;
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            use libc::open64;

            use crate::sys::os::errno;
            let pfds: &mut [_] = &mut [
                libc::pollfd { fd: 0, events: 0, revents: 0 },
                libc::pollfd { fd: 1, events: 0, revents: 0 },
                libc::pollfd { fd: 2, events: 0, revents: 0 },
            ];

            while libc::poll(pfds.as_mut_ptr(), 3, 0) == -1 {
                match errno() {
                    libc::EINTR => continue,
                    #[cfg(target_vendor = "unikraft")]
                    libc::ENOSYS => {
                        // Not all configurations of Unikraft enable `LIBPOSIX_EVENT`.
                        break 'poll;
                    }
                    libc::EINVAL | libc::EAGAIN | libc::ENOMEM => {
                        // RLIMIT_NOFILE or temporary allocation failures
                        // may be preventing use of poll(), fall back to fcntl
                        break 'poll;
                    }
                    _ => libc::abort(),
                }
            }
            for pfd in pfds {
                if pfd.revents & libc::POLLNVAL == 0 {
                    continue;
                }
                if open64(c"/dev/null".as_ptr(), libc::O_RDWR, 0) == -1 {
                    // If the stream is closed but we failed to reopen it, abort the
                    // process. Otherwise we wouldn't preserve the safety of
                    // operations on the corresponding Rust object Stdin, Stdout, or
                    // Stderr.
                    libc::abort();
                }
            }
            return;
        }

        // fallback in case poll isn't available or limited by RLIMIT_NOFILE
        #[cfg(not(any(
            // The standard fds are always available in Miri.
            miri,
            target_os = "emscripten",
            target_os = "fuchsia",
            target_os = "vxworks",
            target_os = "l4re",
            target_os = "horizon",
            target_os = "vita",
        )))]
        {
            #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
            use libc::open as open64;
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            use libc::open64;

            use crate::sys::os::errno;
            for fd in 0..3 {
                if libc::fcntl(fd, libc::F_GETFD) == -1 && errno() == libc::EBADF {
                    if open64(c"/dev/null".as_ptr(), libc::O_RDWR, 0) == -1 {
                        // If the stream is closed but we failed to reopen it, abort the
                        // process. Otherwise we wouldn't preserve the safety of
                        // operations on the corresponding Rust object Stdin, Stdout, or
                        // Stderr.
                        libc::abort();
                    }
                }
            }
        }
    }

    unsafe fn reset_sigpipe(#[allow(unused_variables)] sigpipe: u8) {
        #[cfg(not(any(
            target_os = "emscripten",
            target_os = "fuchsia",
            target_os = "horizon",
            target_os = "vxworks",
            target_os = "vita",
            // Unikraft's `signal` implementation is currently broken:
            // https://github.com/unikraft/lib-musl/issues/57
            target_vendor = "unikraft",
        )))]
        {
            // We don't want to add this as a public type to std, nor do we
            // want to `include!` a file from the compiler (which would break
            // Miri and xargo for example), so we choose to duplicate these
            // constants from `compiler/rustc_session/src/config/sigpipe.rs`.
            // See the other file for docs. NOTE: Make sure to keep them in
            // sync!
            mod sigpipe {
                pub const DEFAULT: u8 = 0;
                pub const INHERIT: u8 = 1;
                pub const SIG_IGN: u8 = 2;
                pub const SIG_DFL: u8 = 3;
            }

            let (sigpipe_attr_specified, handler) = match sigpipe {
                sigpipe::DEFAULT => (false, Some(libc::SIG_IGN)),
                sigpipe::INHERIT => (true, None),
                sigpipe::SIG_IGN => (true, Some(libc::SIG_IGN)),
                sigpipe::SIG_DFL => (true, Some(libc::SIG_DFL)),
                _ => unreachable!(),
            };
            if sigpipe_attr_specified {
                ON_BROKEN_PIPE_FLAG_USED.store(true, crate::sync::atomic::Ordering::Relaxed);
            }
            if let Some(handler) = handler {
                rtassert!(signal(libc::SIGPIPE, handler) != libc::SIG_ERR);
                #[cfg(target_os = "hurd")]
                {
                    rtassert!(signal(libc::SIGLOST, handler) != libc::SIG_ERR);
                }
            }
        }
    }
}

// This is set (up to once) in reset_sigpipe.
#[cfg(not(any(
    target_os = "espidf",
    target_os = "emscripten",
    target_os = "fuchsia",
    target_os = "horizon",
    target_os = "vxworks",
    target_os = "vita",
)))]
static ON_BROKEN_PIPE_FLAG_USED: crate::sync::atomic::AtomicBool =
    crate::sync::atomic::AtomicBool::new(false);

#[cfg(not(any(
    target_os = "espidf",
    target_os = "emscripten",
    target_os = "fuchsia",
    target_os = "horizon",
    target_os = "vxworks",
    target_os = "vita",
    target_os = "nuttx",
)))]
pub(crate) fn on_broken_pipe_flag_used() -> bool {
    ON_BROKEN_PIPE_FLAG_USED.load(crate::sync::atomic::Ordering::Relaxed)
}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {
    stack_overflow::cleanup();
}

#[allow(unused_imports)]
pub use libc::signal;

#[inline]
pub(crate) fn is_interrupted(errno: i32) -> bool {
    errno == libc::EINTR
}

pub fn decode_error_kind(errno: i32) -> ErrorKind {
    use ErrorKind::*;
    match errno as libc::c_int {
        libc::E2BIG => ArgumentListTooLong,
        libc::EADDRINUSE => AddrInUse,
        libc::EADDRNOTAVAIL => AddrNotAvailable,
        libc::EBUSY => ResourceBusy,
        libc::ECONNABORTED => ConnectionAborted,
        libc::ECONNREFUSED => ConnectionRefused,
        libc::ECONNRESET => ConnectionReset,
        libc::EDEADLK => Deadlock,
        libc::EDQUOT => QuotaExceeded,
        libc::EEXIST => AlreadyExists,
        libc::EFBIG => FileTooLarge,
        libc::EHOSTUNREACH => HostUnreachable,
        libc::EINTR => Interrupted,
        libc::EINVAL => InvalidInput,
        libc::EISDIR => IsADirectory,
        libc::ELOOP => FilesystemLoop,
        libc::ENOENT => NotFound,
        libc::ENOMEM => OutOfMemory,
        libc::ENOSPC => StorageFull,
        libc::ENOSYS => Unsupported,
        libc::EMLINK => TooManyLinks,
        libc::ENAMETOOLONG => InvalidFilename,
        libc::ENETDOWN => NetworkDown,
        libc::ENETUNREACH => NetworkUnreachable,
        libc::ENOTCONN => NotConnected,
        libc::ENOTDIR => NotADirectory,
        #[cfg(not(target_os = "aix"))]
        libc::ENOTEMPTY => DirectoryNotEmpty,
        libc::EPIPE => BrokenPipe,
        libc::EROFS => ReadOnlyFilesystem,
        libc::ESPIPE => NotSeekable,
        libc::ESTALE => StaleNetworkFileHandle,
        libc::ETIMEDOUT => TimedOut,
        libc::ETXTBSY => ExecutableFileBusy,
        libc::EXDEV => CrossesDevices,
        libc::EINPROGRESS => InProgress,
        libc::EOPNOTSUPP => Unsupported,

        libc::EACCES | libc::EPERM => PermissionDenied,

        // These two constants can have the same value on some systems,
        // but different values on others, so we can't use a match
        // clause
        x if x == libc::EAGAIN || x == libc::EWOULDBLOCK => WouldBlock,

        _ => Uncategorized,
    }
}

#[doc(hidden)]
pub trait IsMinusOne {
    fn is_minus_one(&self) -> bool;
}

macro_rules! impl_is_minus_one {
    ($($t:ident)*) => ($(impl IsMinusOne for $t {
        fn is_minus_one(&self) -> bool {
            *self == -1
        }
    })*)
}

impl_is_minus_one! { i8 i16 i32 i64 isize }

/// Converts native return values to Result using the *-1 means error is in `errno`*  convention.
/// Non-error values are `Ok`-wrapped.
pub fn cvt<T: IsMinusOne>(t: T) -> crate::io::Result<T> {
    if t.is_minus_one() { Err(crate::io::Error::last_os_error()) } else { Ok(t) }
}

/// `-1` → look at `errno` → retry on `EINTR`. Otherwise `Ok()`-wrap the closure return value.
pub fn cvt_r<T, F>(mut f: F) -> crate::io::Result<T>
where
    T: IsMinusOne,
    F: FnMut() -> T,
{
    loop {
        match cvt(f()) {
            Err(ref e) if e.is_interrupted() => {}
            other => return other,
        }
    }
}

#[allow(dead_code)] // Not used on all platforms.
/// Zero means `Ok()`, all other values are treated as raw OS errors. Does not look at `errno`.
pub fn cvt_nz(error: libc::c_int) -> crate::io::Result<()> {
    if error == 0 { Ok(()) } else { Err(crate::io::Error::from_raw_os_error(error)) }
}

// libc::abort() will run the SIGABRT handler.  That's fine because anyone who
// installs a SIGABRT handler already has to expect it to run in Very Bad
// situations (eg, malloc crashing).
//
// Current glibc's abort() function unblocks SIGABRT, raises SIGABRT, clears the
// SIGABRT handler and raises it again, and then starts to get creative.
//
// See the public documentation for `intrinsics::abort()` and `process::abort()`
// for further discussion.
//
// There is confusion about whether libc::abort() flushes stdio streams.
// libc::abort() is required by ISO C 99 (7.14.1.1p5) to be async-signal-safe,
// so flushing streams is at least extremely hard, if not entirely impossible.
//
// However, some versions of POSIX (eg IEEE Std 1003.1-2001) required abort to
// do so.  In 1003.1-2004 this was fixed.
//
// glibc's implementation did the flush, unsafely, before glibc commit
// 91e7cf982d01 `abort: Do not flush stdio streams [BZ #15436]` by Florian
// Weimer.  According to glibc's NEWS:
//
//    The abort function terminates the process immediately, without flushing
//    stdio streams.  Previous glibc versions used to flush streams, resulting
//    in deadlocks and further data corruption.  This change also affects
//    process aborts as the result of assertion failures.
//
// This is an accurate description of the problem.  The only solution for
// program with nontrivial use of C stdio is a fixed libc - one which does not
// try to flush in abort - since even libc-internal errors, and assertion
// failures generated from C, will go via abort().
//
// On systems with old, buggy, libcs, the impact can be severe for a
// multithreaded C program.  It is much less severe for Rust, because Rust
// stdlib doesn't use libc stdio buffering.  In a typical Rust program, which
// does not use C stdio, even a buggy libc::abort() is, in fact, safe.
pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}

cfg_if::cfg_if! {
    if #[cfg(target_os = "android")] {
        #[link(name = "dl", kind = "static", modifiers = "-bundle",
            cfg(target_feature = "crt-static"))]
        #[link(name = "dl", cfg(not(target_feature = "crt-static")))]
        #[link(name = "log", cfg(not(target_feature = "crt-static")))]
        unsafe extern "C" {}
    } else if #[cfg(target_os = "freebsd")] {
        #[link(name = "execinfo")]
        #[link(name = "pthread")]
        unsafe extern "C" {}
    } else if #[cfg(target_os = "netbsd")] {
        #[link(name = "pthread")]
        #[link(name = "rt")]
        unsafe extern "C" {}
    } else if #[cfg(any(target_os = "dragonfly", target_os = "openbsd", target_os = "cygwin"))] {
        #[link(name = "pthread")]
        unsafe extern "C" {}
    } else if #[cfg(target_os = "solaris")] {
        #[link(name = "socket")]
        #[link(name = "posix4")]
        #[link(name = "pthread")]
        #[link(name = "resolv")]
        unsafe extern "C" {}
    } else if #[cfg(target_os = "illumos")] {
        #[link(name = "socket")]
        #[link(name = "posix4")]
        #[link(name = "pthread")]
        #[link(name = "resolv")]
        #[link(name = "nsl")]
        // Use libumem for the (malloc-compatible) allocator
        #[link(name = "umem")]
        unsafe extern "C" {}
    } else if #[cfg(target_vendor = "apple")] {
        // Link to `libSystem.dylib`.
        //
        // Don't get confused by the presence of `System.framework`,
        // it is a deprecated wrapper over the dynamic library.
        #[link(name = "System")]
        unsafe extern "C" {}
    } else if #[cfg(target_os = "fuchsia")] {
        #[link(name = "zircon")]
        #[link(name = "fdio")]
        unsafe extern "C" {}
    } else if #[cfg(all(target_os = "linux", target_env = "uclibc"))] {
        #[link(name = "dl")]
        unsafe extern "C" {}
    } else if #[cfg(target_os = "vita")] {
        #[link(name = "pthread", kind = "static", modifiers = "-bundle")]
        unsafe extern "C" {}
    }
}

#[cfg(any(target_os = "espidf", target_os = "horizon", target_os = "vita", target_os = "nuttx"))]
pub mod unsupported {
    use crate::io;

    pub fn unsupported<T>() -> io::Result<T> {
        Err(unsupported_err())
    }

    pub fn unsupported_err() -> io::Error {
        io::Error::UNSUPPORTED_PLATFORM
    }
}
