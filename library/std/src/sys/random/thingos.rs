//! ThingOS random number generation.
//!
//! Uses the `SYS_GETRANDOM` system call.  If the syscall is not available
//! (indicated by `-ENOSYS`), the implementation panics, as a quality random
//! source is required for security-critical operations.

use crate::sys::pal::common::{SYS_GETRANDOM, raw_syscall6};

pub fn fill_bytes(buf: &mut [u8]) {
    let mut written = 0usize;
    while written < buf.len() {
        let ret = unsafe {
            raw_syscall6(
                SYS_GETRANDOM,
                buf.as_mut_ptr().add(written) as u64,
                (buf.len() - written) as u64,
                0, // flags: no GRND_NONBLOCK, no GRND_RANDOM
                0,
                0,
                0,
            )
        };
        if ret < 0 {
            panic!("ThingOS SYS_GETRANDOM failed with errno {}", -ret);
        }
        written += ret as usize;
    }
}
