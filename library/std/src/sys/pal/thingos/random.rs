//! Random data for ThingOS via the SYS_GETRANDOM syscall (0x7001).
//!
//! The kernel entropy pool is seeded from hardware (RDRAND/RDSEED) and
//! caps each call at 256 bytes. We retry interruptible calls and briefly yield
//! if early boot entropy is not yet available.
use crate::io;
use crate::sys::pal::raw_syscall6;

/// Syscall number for SYS_GETRANDOM (defined in `abi/src/numbers.rs`).
const SYS_GETRANDOM: u32 = 0x7001;
/// Syscall number for yielding the current task back to the scheduler.
const SYS_YIELD: u32 = 0x100B;
/// Maximum bytes the kernel fills per SYS_GETRANDOM call.
const GETRANDOM_MAX: usize = 256;
/// Maximum number of scheduler yields before surfacing EAGAIN to the caller.
const GETRANDOM_EAGAIN_RETRIES: usize = 1024;

const EINTR: i32 = 4;
const EAGAIN: i32 = 11;

fn fill_bytes_impl<F, Y>(bytes: &mut [u8], mut getrandom: F, mut sched_yield: Y) -> io::Result<()>
where
    F: FnMut(&mut [u8]) -> isize,
    Y: FnMut() -> isize,
{
    let mut offset = 0;
    while offset < bytes.len() {
        let chunk_len = (bytes.len() - offset).min(GETRANDOM_MAX);
        let chunk = &mut bytes[offset..offset + chunk_len];
        let mut eagain_retries = 0;

        loop {
            let ret = getrandom(chunk);
            if ret >= 0 {
                offset += chunk_len;
                break;
            }

            let err = (-ret) as i32;
            match err {
                EINTR => continue,
                EAGAIN if eagain_retries < GETRANDOM_EAGAIN_RETRIES => {
                    eagain_retries += 1;
                    let yield_ret = sched_yield();
                    if yield_ret < 0 {
                        let yield_err = (-yield_ret) as i32;
                        if yield_err != EINTR {
                            return Err(io::Error::from_raw_os_error(yield_err));
                        }
                    }
                }
                _ => return Err(io::Error::from_raw_os_error(err)),
            }
        }
    }

    Ok(())
}

pub fn fill_bytes(bytes: &mut [u8]) -> io::Result<()> {
    fill_bytes_impl(
        bytes,
        |chunk| unsafe {
            raw_syscall6(SYS_GETRANDOM, chunk.as_mut_ptr() as usize, chunk.len(), 0, 0, 0, 0)
        },
        || unsafe { raw_syscall6(SYS_YIELD, 0, 0, 0, 0, 0, 0) },
    )
}

#[cfg(test)]
mod tests {
    use super::fill_bytes_impl;

    const EAGAIN: i32 = 11;
    const EINTR: i32 = 4;

    #[test]
    fn retries_eintr_and_eagain_before_success() {
        let mut buf = [0u8; 8];
        let mut attempts = 0;
        let mut yields = 0;

        let result = fill_bytes_impl(
            &mut buf,
            |_| {
                attempts += 1;
                match attempts {
                    1 => -(EINTR as isize),
                    2 | 3 => -(EAGAIN as isize),
                    _ => 0,
                }
            },
            || {
                yields += 1;
                0
            },
        );

        assert!(result.is_ok());
        assert_eq!(attempts, 4);
        assert_eq!(yields, 2);
    }

    #[test]
    fn returns_eagain_after_retry_budget_exhausted() {
        let mut buf = [0u8; 8];
        let err = fill_bytes_impl(&mut buf, |_| -(EAGAIN as isize), || 0).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(EAGAIN));
    }

    #[test]
    fn returns_other_os_errors_without_panicking() {
        let mut buf = [0u8; 8];
        let err = fill_bytes_impl(&mut buf, |_| -22, || 0).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(22));
    }
}
