#![cfg(any(target_os = "linux", target_os = "android"))]

use crate::sync::atomic::AtomicU32;
use crate::sys::cvt;
use crate::{io, ptr};

pub const fn unlocked() -> u32 {
    0
}

pub fn locked() -> u32 {
    (unsafe { libc::gettid() }) as _
}

pub fn is_contended(futex_val: u32) -> bool {
    (futex_val & libc::FUTEX_WAITERS) != 0
}

pub fn is_owned_died(futex_val: u32) -> bool {
    (futex_val & libc::FUTEX_OWNER_DIED) != 0
}

pub fn futex_lock(futex: &AtomicU32) -> io::Result<()> {
    loop {
        match cvt(unsafe {
            libc::syscall(
                libc::SYS_futex,
                ptr::from_ref(futex),
                libc::FUTEX_LOCK_PI | libc::FUTEX_PRIVATE_FLAG,
                0,
                ptr::null::<u32>(),
                // remaining args are unused
            )
        }) {
            Ok(_) => return Ok(()),
            Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
            Err(e) => return Err(e),
        }
    }
}

pub fn futex_unlock(futex: &AtomicU32) -> io::Result<()> {
    cvt(unsafe {
        libc::syscall(
            libc::SYS_futex,
            ptr::from_ref(futex),
            libc::FUTEX_UNLOCK_PI | libc::FUTEX_PRIVATE_FLAG,
            // remaining args are unused
        )
    })
    .map(|_| ())
}
