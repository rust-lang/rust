//! System bindings for the wasm/web platform
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for wasm. Note that this wasm is *not* the emscripten
//! wasm, so we have no runtime here.
//!
//! This is all super highly experimental and not actually intended for
//! wide/production use yet, it's still all in the experimental category. This
//! will likely change over time.
//!
//! Currently all functions here are basically stubs that immediately return
//! errors. The hope is that with a portability lint we can turn actually just
//! remove all this and just omit parts of the standard library if we're
//! compiling for wasm. That way it's a compile time error for something that's
//! guaranteed to be a runtime error!

use crate::io as std_io;
use crate::mem;

#[path = "../unix/alloc.rs"]
pub mod alloc;
pub mod args;
pub mod env;
pub mod fd;
pub mod fs;
#[allow(unused)]
#[path = "../wasm/atomics/futex.rs"]
pub mod futex;
pub mod io;

pub mod net;
pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
pub mod stdio;
pub mod thread;
#[path = "../unsupported/thread_local_dtor.rs"]
pub mod thread_local_dtor;
#[path = "../unsupported/thread_local_key.rs"]
pub mod thread_local_key;
pub mod time;

cfg_if::cfg_if! {
    if #[cfg(not(target_feature = "atomics"))] {
        #[path = "../unsupported/once.rs"]
        pub mod once;
        #[path = "../unsupported/thread_parking.rs"]
        pub mod thread_parking;
    }
}

#[path = "../unsupported/common.rs"]
#[deny(unsafe_op_in_unsafe_fn)]
#[allow(unused)]
mod common;
pub use common::*;

#[inline]
pub fn is_interrupted(errno: i32) -> bool {
    errno == wasi::ERRNO_INTR.raw().into()
}

pub fn decode_error_kind(errno: i32) -> std_io::ErrorKind {
    use std_io::ErrorKind;

    let Ok(errno) = u16::try_from(errno) else {
        return ErrorKind::Uncategorized;
    };

    macro_rules! match_errno {
        ($($($errno:ident)|+ => $errkind:ident),*, _ => $wildcard:ident $(,)?) => {
            match errno {
                $(e if $(e == ::wasi::$errno.raw())||+ => ErrorKind::$errkind),*,
                _ => ErrorKind::$wildcard,
            }
        };
    }

    match_errno! {
        ERRNO_2BIG           => ArgumentListTooLong,
        ERRNO_ACCES          => PermissionDenied,
        ERRNO_ADDRINUSE      => AddrInUse,
        ERRNO_ADDRNOTAVAIL   => AddrNotAvailable,
        ERRNO_AFNOSUPPORT    => Unsupported,
        ERRNO_AGAIN          => WouldBlock,
        //    ALREADY        => "connection already in progress",
        //    BADF           => "bad file descriptor",
        //    BADMSG         => "bad message",
        ERRNO_BUSY           => ResourceBusy,
        //    CANCELED       => "operation canceled",
        //    CHILD          => "no child processes",
        ERRNO_CONNABORTED    => ConnectionAborted,
        ERRNO_CONNREFUSED    => ConnectionRefused,
        ERRNO_CONNRESET      => ConnectionReset,
        ERRNO_DEADLK         => Deadlock,
        //    DESTADDRREQ    => "destination address required",
        ERRNO_DOM            => InvalidInput,
        //    DQUOT          => /* reserved */,
        ERRNO_EXIST          => AlreadyExists,
        //    FAULT          => "bad address",
        ERRNO_FBIG           => FileTooLarge,
        ERRNO_HOSTUNREACH    => HostUnreachable,
        //    IDRM           => "identifier removed",
        //    ILSEQ          => "illegal byte sequence",
        //    INPROGRESS     => "operation in progress",
        ERRNO_INTR           => Interrupted,
        ERRNO_INVAL          => InvalidInput,
        ERRNO_IO             => Uncategorized,
        //    ISCONN         => "socket is connected",
        ERRNO_ISDIR          => IsADirectory,
        ERRNO_LOOP           => FilesystemLoop,
        //    MFILE          => "file descriptor value too large",
        ERRNO_MLINK          => TooManyLinks,
        //    MSGSIZE        => "message too large",
        //    MULTIHOP       => /* reserved */,
        ERRNO_NAMETOOLONG    => InvalidFilename,
        ERRNO_NETDOWN        => NetworkDown,
        //    NETRESET       => "connection aborted by network",
        ERRNO_NETUNREACH     => NetworkUnreachable,
        //    NFILE          => "too many files open in system",
        //    NOBUFS         => "no buffer space available",
        ERRNO_NODEV          => NotFound,
        ERRNO_NOENT          => NotFound,
        //    NOEXEC         => "executable file format error",
        //    NOLCK          => "no locks available",
        //    NOLINK         => /* reserved */,
        ERRNO_NOMEM          => OutOfMemory,
        //    NOMSG          => "no message of the desired type",
        //    NOPROTOOPT     => "protocol not available",
        ERRNO_NOSPC          => StorageFull,
        ERRNO_NOSYS          => Unsupported,
        ERRNO_NOTCONN        => NotConnected,
        ERRNO_NOTDIR         => NotADirectory,
        ERRNO_NOTEMPTY       => DirectoryNotEmpty,
        //    NOTRECOVERABLE => "state not recoverable",
        //    NOTSOCK        => "not a socket",
        ERRNO_NOTSUP         => Unsupported,
        //    NOTTY          => "inappropriate I/O control operation",
        ERRNO_NXIO           => NotFound,
        //    OVERFLOW       => "value too large to be stored in data type",
        //    OWNERDEAD      => "previous owner died",
        ERRNO_PERM           => PermissionDenied,
        ERRNO_PIPE           => BrokenPipe,
        //    PROTO          => "protocol error",
        ERRNO_PROTONOSUPPORT => Unsupported,
        //    PROTOTYPE      => "protocol wrong type for socket",
        //    RANGE          => "result too large",
        ERRNO_ROFS           => ReadOnlyFilesystem,
        ERRNO_SPIPE          => NotSeekable,
        ERRNO_SRCH           => NotFound,
        //    STALE          => /* reserved */,
        ERRNO_TIMEDOUT       => TimedOut,
        ERRNO_TXTBSY         => ResourceBusy,
        ERRNO_XDEV           => CrossesDevices,
        ERRNO_NOTCAPABLE     => PermissionDenied,
        _                    => Uncategorized,
    }
}

pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}

pub fn hashmap_random_keys() -> (u64, u64) {
    let mut ret = (0u64, 0u64);
    unsafe {
        let base = core::ptr::addr_of_mut!(ret) as *mut u8;
        let len = mem::size_of_val(&ret);
        wasi::random_get(base, len).expect("random_get failure");
    }
    return ret;
}

#[inline]
fn err2io(err: wasi::Errno) -> std_io::Error {
    std_io::Error::from_raw_os_error(err.raw().into())
}
