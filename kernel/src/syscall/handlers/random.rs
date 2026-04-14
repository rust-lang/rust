//! Random/entropy syscall handlers.

use crate::syscall::validate::{copyin, copyout, validate_user_range};
use abi::errors::SysResult;

/// SYS_GETRANDOM: Fill a user buffer with random bytes.
///
/// Args: buf_ptr, buf_len.
/// Returns: 0 on success.
pub fn sys_getrandom(buf_ptr: usize, buf_len: usize) -> SysResult<usize> {
    if buf_len == 0 {
        return Ok(0);
    }

    // Cap at 256 bytes per call to avoid holding kernel resources too long
    let len = buf_len.min(256);
    validate_user_range(buf_ptr, len, true)?;

    let mut kbuf = [0u8; 256];
    crate::entropy::fill(&mut kbuf[..len])?;

    unsafe {
        copyout(buf_ptr, &kbuf[..len])?;
    }
    Ok(0)
}

/// SYS_ENTROPY_SEED: Mix caller-supplied bytes into the kernel entropy pool.
///
/// Intended for privileged entropy-source drivers (analogous to `SYS_TIME_ANCHOR`
/// for the system clock). After this call the pool is marked as seeded even if
/// it was not yet seeded from hardware.
///
/// Args: buf_ptr, buf_len (capped at 256 bytes per call).
/// Returns: 0 on success.
pub fn sys_entropy_seed(buf_ptr: usize, buf_len: usize) -> SysResult<usize> {
    if buf_len == 0 {
        return Ok(0);
    }

    let len = buf_len.min(256);
    validate_user_range(buf_ptr, len, false)?;

    let mut kbuf = [0u8; 256];
    unsafe {
        copyin(&mut kbuf[..len], buf_ptr)?;
    }

    crate::entropy::add_sample(&kbuf[..len]);
    crate::entropy::mark_seeded();
    crate::kdebug!("ENTROPY: seeded {} bytes from userspace driver", len);
    Ok(0)
}
