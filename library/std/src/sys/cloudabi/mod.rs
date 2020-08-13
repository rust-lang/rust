#![deny(unsafe_op_in_unsafe_fn)]

use crate::io::ErrorKind;
use crate::mem;

#[path = "../unix/alloc.rs"]
pub mod alloc;
pub mod args;
#[path = "../unix/cmath.rs"]
pub mod cmath;
pub mod condvar;
pub mod io;
#[path = "../unix/memchr.rs"]
pub mod memchr;
pub mod mutex;
pub mod os;
pub mod rwlock;
pub mod stack_overflow;
pub mod stdio;
pub mod thread;
#[path = "../unix/thread_local_key.rs"]
pub mod thread_local_key;
pub mod time;

pub use crate::sys_common::os_str_bytes as os_str;

mod abi;

mod shims;
pub use self::shims::*;

#[allow(dead_code)]
pub fn init() {}

pub fn decode_error_kind(errno: i32) -> ErrorKind {
    match errno {
        x if x == abi::errno::ACCES as i32 => ErrorKind::PermissionDenied,
        x if x == abi::errno::ADDRINUSE as i32 => ErrorKind::AddrInUse,
        x if x == abi::errno::ADDRNOTAVAIL as i32 => ErrorKind::AddrNotAvailable,
        x if x == abi::errno::AGAIN as i32 => ErrorKind::WouldBlock,
        x if x == abi::errno::CONNABORTED as i32 => ErrorKind::ConnectionAborted,
        x if x == abi::errno::CONNREFUSED as i32 => ErrorKind::ConnectionRefused,
        x if x == abi::errno::CONNRESET as i32 => ErrorKind::ConnectionReset,
        x if x == abi::errno::EXIST as i32 => ErrorKind::AlreadyExists,
        x if x == abi::errno::INTR as i32 => ErrorKind::Interrupted,
        x if x == abi::errno::INVAL as i32 => ErrorKind::InvalidInput,
        x if x == abi::errno::NOENT as i32 => ErrorKind::NotFound,
        x if x == abi::errno::NOTCONN as i32 => ErrorKind::NotConnected,
        x if x == abi::errno::PERM as i32 => ErrorKind::PermissionDenied,
        x if x == abi::errno::PIPE as i32 => ErrorKind::BrokenPipe,
        x if x == abi::errno::TIMEDOUT as i32 => ErrorKind::TimedOut,
        _ => ErrorKind::Other,
    }
}

pub fn abort_internal() -> ! {
    core::intrinsics::abort();
}

pub use libc::strlen;

pub fn hashmap_random_keys() -> (u64, u64) {
    unsafe {
        let mut v: mem::MaybeUninit<(u64, u64)> = mem::MaybeUninit::uninit();
        libc::arc4random_buf(v.as_mut_ptr() as *mut libc::c_void, mem::size_of_val(&v));
        v.assume_init()
    }
}
