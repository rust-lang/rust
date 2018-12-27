use io;
use libc;
use mem;

#[path = "../unix/alloc.rs"]
pub mod alloc;
pub mod args;
#[cfg(feature = "backtrace")]
pub mod backtrace;
#[path = "../unix/cmath.rs"]
pub mod cmath;
pub mod condvar;
#[path = "../unix/memchr.rs"]
pub mod memchr;
pub mod mutex;
pub mod os;
#[path = "../unix/os_str.rs"]
pub mod os_str;
pub mod rwlock;
pub mod stack_overflow;
pub mod stdio;
pub mod thread;
#[path = "../unix/thread_local.rs"]
pub mod thread_local;
pub mod time;

mod abi;

mod shims;
pub use self::shims::*;

#[allow(dead_code)]
pub fn init() {}

pub fn decode_error_kind(errno: i32) -> io::ErrorKind {
    match errno {
        x if x == abi::errno::ACCES as i32 => io::ErrorKind::PermissionDenied,
        x if x == abi::errno::ADDRINUSE as i32 => io::ErrorKind::AddrInUse,
        x if x == abi::errno::ADDRNOTAVAIL as i32 => io::ErrorKind::AddrNotAvailable,
        x if x == abi::errno::AGAIN as i32 => io::ErrorKind::WouldBlock,
        x if x == abi::errno::CONNABORTED as i32 => io::ErrorKind::ConnectionAborted,
        x if x == abi::errno::CONNREFUSED as i32 => io::ErrorKind::ConnectionRefused,
        x if x == abi::errno::CONNRESET as i32 => io::ErrorKind::ConnectionReset,
        x if x == abi::errno::EXIST as i32 => io::ErrorKind::AlreadyExists,
        x if x == abi::errno::INTR as i32 => io::ErrorKind::Interrupted,
        x if x == abi::errno::INVAL as i32 => io::ErrorKind::InvalidInput,
        x if x == abi::errno::NOENT as i32 => io::ErrorKind::NotFound,
        x if x == abi::errno::NOTCONN as i32 => io::ErrorKind::NotConnected,
        x if x == abi::errno::PERM as i32 => io::ErrorKind::PermissionDenied,
        x if x == abi::errno::PIPE as i32 => io::ErrorKind::BrokenPipe,
        x if x == abi::errno::TIMEDOUT as i32 => io::ErrorKind::TimedOut,
        _ => io::ErrorKind::Other,
    }
}

pub unsafe fn abort_internal() -> ! {
    ::core::intrinsics::abort();
}

pub use libc::strlen;

pub fn hashmap_random_keys() -> (u64, u64) {
    unsafe {
        let mut v = mem::uninitialized();
        libc::arc4random_buf(&mut v as *mut _ as *mut libc::c_void, mem::size_of_val(&v));
        v
    }
}
