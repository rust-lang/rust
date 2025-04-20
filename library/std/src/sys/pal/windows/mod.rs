#![allow(missing_docs, nonstandard_style)]
#![forbid(unsafe_op_in_unsafe_fn)]

use crate::ffi::{OsStr, OsString};
use crate::io::ErrorKind;
use crate::mem::MaybeUninit;
use crate::os::windows::ffi::{OsStrExt, OsStringExt};
use crate::path::PathBuf;
use crate::sys::pal::windows::api::wide_str;
use crate::time::Duration;

#[macro_use]
pub mod compat;

pub mod api;

pub mod c;
#[cfg(not(target_vendor = "win7"))]
pub mod futex;
pub mod handle;
pub mod os;
pub mod pipe;
pub mod thread;
pub mod time;
cfg_if::cfg_if! {
    if #[cfg(not(target_vendor = "uwp"))] {
        pub mod stack_overflow;
    } else {
        pub mod stack_overflow_uwp;
        pub use self::stack_overflow_uwp as stack_overflow;
    }
}

/// Map a [`Result<T, WinError>`] to [`io::Result<T>`](crate::io::Result<T>).
pub trait IoResult<T> {
    fn io_result(self) -> crate::io::Result<T>;
}
impl<T> IoResult<T> for Result<T, api::WinError> {
    fn io_result(self) -> crate::io::Result<T> {
        self.map_err(|e| crate::io::Error::from_raw_os_error(e.code as i32))
    }
}

// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
pub unsafe fn init(_argc: isize, _argv: *const *const u8, _sigpipe: u8) {
    unsafe {
        stack_overflow::init();

        // Normally, `thread::spawn` will call `Thread::set_name` but since this thread already
        // exists, we have to call it ourselves.
        thread::Thread::set_name_wide(wide_str!("main"));
    }
}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {
    crate::sys::net::cleanup();
}

#[inline]
pub fn is_interrupted(_errno: i32) -> bool {
    false
}

pub fn decode_error_kind(errno: i32) -> ErrorKind {
    use ErrorKind::*;

    match errno as u32 {
        c::ERROR_ACCESS_DENIED => return PermissionDenied,
        c::ERROR_ALREADY_EXISTS => return AlreadyExists,
        c::ERROR_FILE_EXISTS => return AlreadyExists,
        c::ERROR_BROKEN_PIPE => return BrokenPipe,
        c::ERROR_FILE_NOT_FOUND
        | c::ERROR_PATH_NOT_FOUND
        | c::ERROR_INVALID_DRIVE
        | c::ERROR_BAD_NETPATH
        | c::ERROR_BAD_NET_NAME => return NotFound,
        c::ERROR_NO_DATA => return BrokenPipe,
        c::ERROR_INVALID_NAME | c::ERROR_BAD_PATHNAME => return InvalidFilename,
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
        c::ERROR_DISK_QUOTA_EXCEEDED => return QuotaExceeded,
        c::ERROR_FILE_TOO_LARGE => return FileTooLarge,
        c::ERROR_BUSY => return ResourceBusy,
        c::ERROR_POSSIBLE_DEADLOCK => return Deadlock,
        c::ERROR_NOT_SAME_DEVICE => return CrossesDevices,
        c::ERROR_TOO_MANY_LINKS => return TooManyLinks,
        c::ERROR_FILENAME_EXCED_RANGE => return InvalidFilename,
        c::ERROR_CANT_RESOLVE_FILENAME => return FilesystemLoop,
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
        c::WSAEDQUOT => QuotaExceeded,

        _ => Uncategorized,
    }
}

pub fn unrolled_find_u16s(needle: u16, haystack: &[u16]) -> Option<usize> {
    let ptr = haystack.as_ptr();
    let mut start = haystack;

    // For performance reasons unfold the loop eight times.
    while start.len() >= 8 {
        macro_rules! if_return {
            ($($n:literal,)+) => {
                $(
                    if start[$n] == needle {
                        return Some(((&start[$n] as *const u16).addr() - ptr.addr()) / 2);
                    }
                )+
            }
        }

        if_return!(0, 1, 2, 3, 4, 5, 6, 7,);

        start = &start[8..];
    }

    for c in start {
        if *c == needle {
            return Some(((c as *const u16).addr() - ptr.addr()) / 2);
        }
    }
    None
}

pub fn to_u16s<S: AsRef<OsStr>>(s: S) -> crate::io::Result<Vec<u16>> {
    fn inner(s: &OsStr) -> crate::io::Result<Vec<u16>> {
        // Most paths are ASCII, so reserve capacity for as much as there are bytes
        // in the OsStr plus one for the null-terminating character. We are not
        // wasting bytes here as paths created by this function are primarily used
        // in an ephemeral fashion.
        let mut maybe_result = Vec::with_capacity(s.len() + 1);
        maybe_result.extend(s.encode_wide());

        if unrolled_find_u16s(0, &maybe_result).is_some() {
            return Err(crate::io::const_error!(
                ErrorKind::InvalidInput,
                "strings passed to WinAPI cannot contain NULs",
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
// The first callback, `f1`, is passed a (pointer, len) pair which can be
// passed to a syscall. The `ptr` is valid for `len` items (u16 in this case).
// The closure is expected to:
// - On success, return the actual length of the written data *without* the null terminator.
//   This can be 0. In this case the last_error must be left unchanged.
// - On insufficient buffer space,
//   - either return the required length *with* the null terminator,
//   - or set the last-error to ERROR_INSUFFICIENT_BUFFER and return `len`.
// - On other failure, return 0 and set last_error.
//
// This is how most but not all syscalls indicate the required buffer space.
// Other syscalls may need translation to match this protocol.
//
// Once the syscall has completed (errors bail out early) the second closure is
// passed the data which has been read from the syscall. The return value
// from this closure is then the return value of the function.
pub fn fill_utf16_buf<F1, F2, T>(mut f1: F1, f2: F2) -> crate::io::Result<T>
where
    F1: FnMut(*mut u16, u32) -> u32,
    F2: FnOnce(&[u16]) -> T,
{
    // Start off with a stack buf but then spill over to the heap if we end up
    // needing more space.
    //
    // This initial size also works around `GetFullPathNameW` returning
    // incorrect size hints for some short paths:
    // https://github.com/dylni/normpath/issues/5
    let mut stack_buf: [MaybeUninit<u16>; 512] = [MaybeUninit::uninit(); 512];
    let mut heap_buf: Vec<MaybeUninit<u16>> = Vec::new();
    unsafe {
        let mut n = stack_buf.len();
        loop {
            let buf = if n <= stack_buf.len() {
                &mut stack_buf[..]
            } else {
                let extra = n - heap_buf.len();
                heap_buf.reserve(extra);
                // We used `reserve` and not `reserve_exact`, so in theory we
                // may have gotten more than requested. If so, we'd like to use
                // it... so long as we won't cause overflow.
                n = heap_buf.capacity().min(u32::MAX as usize);
                // Safety: MaybeUninit<u16> does not need initialization
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
            let k = match f1(buf.as_mut_ptr().cast::<u16>(), n as u32) {
                0 if api::get_last_error().code == 0 => 0,
                0 => return Err(crate::io::Error::last_os_error()),
                n => n,
            } as usize;
            if k == n && api::get_last_error().code == c::ERROR_INSUFFICIENT_BUFFER {
                n = n.saturating_mul(2).min(u32::MAX as usize);
            } else if k > n {
                n = k;
            } else if k == n {
                // It is impossible to reach this point.
                // On success, k is the returned string length excluding the null.
                // On failure, k is the required buffer length including the null.
                // Therefore k never equals n.
                unreachable!();
            } else {
                // Safety: First `k` values are initialized.
                let slice: &[u16] = buf[..k].assume_init_ref();
                return Ok(f2(slice));
            }
        }
    }
}

pub fn os2path(s: &[u16]) -> PathBuf {
    PathBuf::from(OsString::from_wide(s))
}

pub fn truncate_utf16_at_nul(v: &[u16]) -> &[u16] {
    match unrolled_find_u16s(0, v) {
        // don't include the 0
        Some(i) => &v[..i],
        None => v,
    }
}

pub fn ensure_no_nuls<T: AsRef<OsStr>>(s: T) -> crate::io::Result<T> {
    if s.as_ref().encode_wide().any(|b| b == 0) {
        Err(crate::io::const_error!(ErrorKind::InvalidInput, "nul byte found in provided data"))
    } else {
        Ok(s)
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

pub fn dur2timeout(dur: Duration) -> u32 {
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
        .map(|ms| if ms > <u32>::MAX as u64 { c::INFINITE } else { ms as u32 })
        .unwrap_or(c::INFINITE)
}

/// Use `__fastfail` to abort the process
///
/// This is the same implementation as in libpanic_abort's `__rust_start_panic`. See
/// that function for more information on `__fastfail`
#[cfg(not(miri))] // inline assembly does not work in Miri
pub fn abort_internal() -> ! {
    unsafe {
        cfg_if::cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                core::arch::asm!("int $$0x29", in("ecx") c::FAST_FAIL_FATAL_APP_EXIT, options(noreturn, nostack));
            } else if #[cfg(all(target_arch = "arm", target_feature = "thumb-mode"))] {
                core::arch::asm!(".inst 0xDEFB", in("r0") c::FAST_FAIL_FATAL_APP_EXIT, options(noreturn, nostack));
            } else if #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))] {
                core::arch::asm!("brk 0xF003", in("x0") c::FAST_FAIL_FATAL_APP_EXIT, options(noreturn, nostack));
            } else {
                core::intrinsics::abort();
            }
        }
    }
}

#[cfg(miri)]
pub fn abort_internal() -> ! {
    crate::intrinsics::abort();
}

/// Align the inner value to 8 bytes.
///
/// This is enough for almost all of the buffers we're likely to work with in
/// the Windows APIs we use.
#[repr(C, align(8))]
#[derive(Copy, Clone)]
pub(crate) struct Align8<T: ?Sized>(pub T);
