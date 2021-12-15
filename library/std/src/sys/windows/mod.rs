#![allow(missing_docs, nonstandard_style)]

use crate::ffi::{OsStr, OsString};
use crate::io::ErrorKind;
use crate::os::windows::ffi::{OsStrExt, OsStringExt};
use crate::path::PathBuf;
use crate::time::Duration;

pub use self::rand::hashmap_random_keys;
pub use libc::strlen;

#[macro_use]
pub mod compat;

pub mod alloc;
pub mod args;
pub mod c;
pub mod cmath;
pub mod condvar;
pub mod env;
pub mod fs;
pub mod handle;
pub mod io;
pub mod memchr;
pub mod mutex;
pub mod net;
pub mod os;
pub mod os_str;
pub mod path;
pub mod pipe;
pub mod process;
pub mod rand;
pub mod rwlock;
pub mod thread;
pub mod thread_local_dtor;
pub mod thread_local_key;
pub mod thread_parker;
pub mod time;
cfg_if::cfg_if! {
    if #[cfg(not(target_vendor = "uwp"))] {
        pub mod stdio;
        pub mod stack_overflow;
    } else {
        pub mod stdio_uwp;
        pub mod stack_overflow_uwp;
        pub use self::stdio_uwp as stdio;
        pub use self::stack_overflow_uwp as stack_overflow;
    }
}

// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
pub unsafe fn init(_argc: isize, _argv: *const *const u8) {
    stack_overflow::init();
}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {
    net::cleanup();
}

pub fn decode_error_kind(errno: i32) -> ErrorKind {
    use ErrorKind::*;

    match errno as c::DWORD {
        c::ERROR_ACCESS_DENIED => return PermissionDenied,
        c::ERROR_ALREADY_EXISTS => return AlreadyExists,
        c::ERROR_FILE_EXISTS => return AlreadyExists,
        c::ERROR_BROKEN_PIPE => return BrokenPipe,
        c::ERROR_FILE_NOT_FOUND => return NotFound,
        c::ERROR_PATH_NOT_FOUND => return NotFound,
        c::ERROR_NO_DATA => return BrokenPipe,
        c::ERROR_INVALID_PARAMETER => return InvalidInput,
        c::ERROR_NOT_ENOUGH_MEMORY | c::ERROR_OUTOFMEMORY => return OutOfMemory,
        c::ERROR_SEM_TIMEOUT
        | c::WAIT_TIMEOUT
        | c::ERROR_DRIVER_CANCEL_TIMEOUT
        | c::ERROR_OPERATION_ABORTED
        | c::ERROR_SERVICE_REQUEST_TIMEOUT
        | c::ERROR_COUNTER_TIMEOUT
        | c::ERROR_TIMEOUT
        | c::ERROR_RESOURCE_CALL_TIMED_OUT
        | c::ERROR_CTX_MODEM_RESPONSE_TIMEOUT
        | c::ERROR_CTX_CLIENT_QUERY_TIMEOUT
        | c::FRS_ERR_SYSVOL_POPULATE_TIMEOUT
        | c::ERROR_DS_TIMELIMIT_EXCEEDED
        | c::DNS_ERROR_RECORD_TIMED_OUT
        | c::ERROR_IPSEC_IKE_TIMED_OUT
        | c::ERROR_RUNLEVEL_SWITCH_TIMEOUT
        | c::ERROR_RUNLEVEL_SWITCH_AGENT_TIMEOUT => return TimedOut,
        c::ERROR_CALL_NOT_IMPLEMENTED => return Unsupported,
        c::ERROR_HOST_UNREACHABLE => return HostUnreachable,
        c::ERROR_NETWORK_UNREACHABLE => return NetworkUnreachable,
        c::ERROR_DIRECTORY => return NotADirectory,
        c::ERROR_DIRECTORY_NOT_SUPPORTED => return IsADirectory,
        c::ERROR_DIR_NOT_EMPTY => return DirectoryNotEmpty,
        c::ERROR_WRITE_PROTECT => return ReadOnlyFilesystem,
        c::ERROR_DISK_FULL | c::ERROR_HANDLE_DISK_FULL => return StorageFull,
        c::ERROR_SEEK_ON_DEVICE => return NotSeekable,
        c::ERROR_DISK_QUOTA_EXCEEDED => return FilesystemQuotaExceeded,
        c::ERROR_FILE_TOO_LARGE => return FileTooLarge,
        c::ERROR_BUSY => return ResourceBusy,
        c::ERROR_POSSIBLE_DEADLOCK => return Deadlock,
        c::ERROR_NOT_SAME_DEVICE => return CrossesDevices,
        c::ERROR_TOO_MANY_LINKS => return TooManyLinks,
        c::ERROR_FILENAME_EXCED_RANGE => return FilenameTooLong,
        _ => {}
    }

    match errno {
        c::WSAEACCES => PermissionDenied,
        c::WSAEADDRINUSE => AddrInUse,
        c::WSAEADDRNOTAVAIL => AddrNotAvailable,
        c::WSAECONNABORTED => ConnectionAborted,
        c::WSAECONNREFUSED => ConnectionRefused,
        c::WSAECONNRESET => ConnectionReset,
        c::WSAEINVAL => InvalidInput,
        c::WSAENOTCONN => NotConnected,
        c::WSAEWOULDBLOCK => WouldBlock,
        c::WSAETIMEDOUT => TimedOut,
        c::WSAEHOSTUNREACH => HostUnreachable,
        c::WSAENETDOWN => NetworkDown,
        c::WSAENETUNREACH => NetworkUnreachable,

        _ => Uncategorized,
    }
}

pub fn unrolled_find_u16s(needle: u16, haystack: &[u16]) -> Option<usize> {
    let ptr = haystack.as_ptr();
    let mut start = &haystack[..];

    // For performance reasons unfold the loop eight times.
    while start.len() >= 8 {
        macro_rules! if_return {
            ($($n:literal,)+) => {
                $(
                    if start[$n] == needle {
                        return Some((&start[$n] as *const u16 as usize - ptr as usize) / 2);
                    }
                )+
            }
        }

        if_return!(0, 1, 2, 3, 4, 5, 6, 7,);

        start = &start[8..];
    }

    for c in start {
        if *c == needle {
            return Some((c as *const u16 as usize - ptr as usize) / 2);
        }
    }
    None
}

pub fn to_u16s<S: AsRef<OsStr>>(s: S) -> crate::io::Result<Vec<u16>> {
    fn inner(s: &OsStr) -> crate::io::Result<Vec<u16>> {
        let mut maybe_result: Vec<u16> = s.encode_wide().collect();
        if unrolled_find_u16s(0, &maybe_result).is_some() {
            return Err(crate::io::Error::new_const(
                ErrorKind::InvalidInput,
                &"strings passed to WinAPI cannot contain NULs",
            ));
        }
        maybe_result.push(0);
        Ok(maybe_result)
    }
    inner(s.as_ref())
}

// Many Windows APIs follow a pattern of where we hand a buffer and then they
// will report back to us how large the buffer should be or how many bytes
// currently reside in the buffer. This function is an abstraction over these
// functions by making them easier to call.
//
// The first callback, `f1`, is yielded a (pointer, len) pair which can be
// passed to a syscall. The `ptr` is valid for `len` items (u16 in this case).
// The closure is expected to return what the syscall returns which will be
// interpreted by this function to determine if the syscall needs to be invoked
// again (with more buffer space).
//
// Once the syscall has completed (errors bail out early) the second closure is
// yielded the data which has been read from the syscall. The return value
// from this closure is then the return value of the function.
fn fill_utf16_buf<F1, F2, T>(mut f1: F1, f2: F2) -> crate::io::Result<T>
where
    F1: FnMut(*mut u16, c::DWORD) -> c::DWORD,
    F2: FnOnce(&[u16]) -> T,
{
    // Start off with a stack buf but then spill over to the heap if we end up
    // needing more space.
    let mut stack_buf = [0u16; 512];
    let mut heap_buf = Vec::new();
    unsafe {
        let mut n = stack_buf.len();
        loop {
            let buf = if n <= stack_buf.len() {
                &mut stack_buf[..]
            } else {
                let extra = n - heap_buf.len();
                heap_buf.reserve(extra);
                heap_buf.set_len(n);
                &mut heap_buf[..]
            };

            // This function is typically called on windows API functions which
            // will return the correct length of the string, but these functions
            // also return the `0` on error. In some cases, however, the
            // returned "correct length" may actually be 0!
            //
            // To handle this case we call `SetLastError` to reset it to 0 and
            // then check it again if we get the "0 error value". If the "last
            // error" is still 0 then we interpret it as a 0 length buffer and
            // not an actual error.
            c::SetLastError(0);
            let k = match f1(buf.as_mut_ptr(), n as c::DWORD) {
                0 if c::GetLastError() == 0 => 0,
                0 => return Err(crate::io::Error::last_os_error()),
                n => n,
            } as usize;
            if k == n && c::GetLastError() == c::ERROR_INSUFFICIENT_BUFFER {
                n *= 2;
            } else if k >= n {
                n = k;
            } else {
                return Ok(f2(&buf[..k]));
            }
        }
    }
}

fn os2path(s: &[u16]) -> PathBuf {
    PathBuf::from(OsString::from_wide(s))
}

pub fn truncate_utf16_at_nul(v: &[u16]) -> &[u16] {
    match unrolled_find_u16s(0, v) {
        // don't include the 0
        Some(i) => &v[..i],
        None => v,
    }
}

pub trait IsZero {
    fn is_zero(&self) -> bool;
}

macro_rules! impl_is_zero {
    ($($t:ident)*) => ($(impl IsZero for $t {
        fn is_zero(&self) -> bool {
            *self == 0
        }
    })*)
}

impl_is_zero! { i8 i16 i32 i64 isize u8 u16 u32 u64 usize }

pub fn cvt<I: IsZero>(i: I) -> crate::io::Result<I> {
    if i.is_zero() { Err(crate::io::Error::last_os_error()) } else { Ok(i) }
}

pub fn dur2timeout(dur: Duration) -> c::DWORD {
    // Note that a duration is a (u64, u32) (seconds, nanoseconds) pair, and the
    // timeouts in windows APIs are typically u32 milliseconds. To translate, we
    // have two pieces to take care of:
    //
    // * Nanosecond precision is rounded up
    // * Greater than u32::MAX milliseconds (50 days) is rounded up to INFINITE
    //   (never time out).
    dur.as_secs()
        .checked_mul(1000)
        .and_then(|ms| ms.checked_add((dur.subsec_nanos() as u64) / 1_000_000))
        .and_then(|ms| ms.checked_add(if dur.subsec_nanos() % 1_000_000 > 0 { 1 } else { 0 }))
        .map(|ms| if ms > <c::DWORD>::MAX as u64 { c::INFINITE } else { ms as c::DWORD })
        .unwrap_or(c::INFINITE)
}

/// Use `__fastfail` to abort the process
///
/// This is the same implementation as in libpanic_abort's `__rust_start_panic`. See
/// that function for more information on `__fastfail`
#[allow(unreachable_code)]
pub fn abort_internal() -> ! {
    const FAST_FAIL_FATAL_APP_EXIT: usize = 7;
    unsafe {
        cfg_if::cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                core::arch::asm!("int $$0x29", in("ecx") FAST_FAIL_FATAL_APP_EXIT);
                crate::intrinsics::unreachable();
            } else if #[cfg(all(target_arch = "arm", target_feature = "thumb-mode"))] {
                core::arch::asm!(".inst 0xDEFB", in("r0") FAST_FAIL_FATAL_APP_EXIT);
                crate::intrinsics::unreachable();
            } else if #[cfg(target_arch = "aarch64")] {
                core::arch::asm!("brk 0xF003", in("x0") FAST_FAIL_FATAL_APP_EXIT);
                crate::intrinsics::unreachable();
            }
        }
    }
    crate::intrinsics::abort();
}
