//! Random data for ThingOS via the SYS_GETRANDOM syscall (0x7001).
//!
//! The kernel entropy pool is seeded from hardware (RDRAND/RDSEED) and
//! caps each call at 256 bytes.  We loop until the caller's buffer is full.
use crate::sys::pal::raw_syscall6;

/// Syscall number for SYS_GETRANDOM (defined in `abi/src/numbers.rs`).
const SYS_GETRANDOM: u32 = 0x7001;
/// Maximum bytes the kernel fills per SYS_GETRANDOM call.
const GETRANDOM_MAX: usize = 256;

pub fn fill_bytes(bytes: &mut [u8]) {
    let mut offset = 0;
    while offset < bytes.len() {
        // Pass at most GETRANDOM_MAX bytes per call so the kernel doesn't
        // reject oversized requests.
        let chunk_len = (bytes.len() - offset).min(GETRANDOM_MAX);
        let chunk = &mut bytes[offset..offset + chunk_len];
        // args: buf_ptr, buf_len, flags (0 = blocking, fill fully)
        let ret = unsafe {
            raw_syscall6(SYS_GETRANDOM, chunk.as_mut_ptr() as usize, chunk_len, 0, 0, 0, 0)
        };
        // SYS_GETRANDOM returns 0 on success (not byte count); a negative
        // value encodes an errno-style error code.
        if ret < 0 {
            panic!("SYS_GETRANDOM failed: {}", ret);
        }
        offset += chunk_len;
    }
}
