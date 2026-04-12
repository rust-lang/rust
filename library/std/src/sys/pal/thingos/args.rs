//! ThingOS `args()` implementation.
//!
//! # ABI
//! `SYS_ARGV_GET(buf_ptr, buf_len)` → total bytes needed.
//!
//! Serialized argv blob format (kernel/src/syscall/handlers/process.rs):
//!   count: u32 LE
//!   for each arg:
//!     len:   u32 LE
//!     bytes: [u8; len]
//!
//! Call with buf_ptr=0/buf_len=0 to query the required size first.

pub use super::common::Args;
use crate::ffi::OsString;

// Syscall number (abi/src/numbers.rs)
const SYS_ARGV_GET: u32 = 0x1100;

#[inline(always)]
unsafe fn raw_syscall6(n: u32, a0: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize) -> isize {
    unsafe { crate::sys::pal::raw_syscall6(n, a0, a1, a2, a3, a4, a5) }
}

/// Retrieve argv from the kernel and return it as an `Args` iterator.
pub fn args() -> Args {
    // Phase 1: query required buffer size.
    let needed = unsafe { raw_syscall6(SYS_ARGV_GET, 0, 0, 0, 0, 0, 0) };
    if needed <= 0 {
        return Args::new(crate::vec![]);
    }
    let needed = needed as usize;

    // Phase 2: fill buffer.
    let mut buf = crate::vec![0u8; needed];
    let ret = unsafe {
        raw_syscall6(SYS_ARGV_GET, buf.as_mut_ptr() as usize, buf.len(), 0, 0, 0, 0)
    };
    if ret < 0 {
        return Args::new(crate::vec![]);
    }

    parse_argv(&buf)
}

fn parse_argv(blob: &[u8]) -> Args {
    if blob.len() < 4 {
        return Args::new(crate::vec![]);
    }
    let count = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
    let mut pos = 4;
    let mut result: crate::vec::Vec<OsString> = crate::vec::Vec::with_capacity(count);

    for _ in 0..count {
        if pos + 4 > blob.len() {
            break;
        }
        let len = u32::from_le_bytes([blob[pos], blob[pos+1], blob[pos+2], blob[pos+3]]) as usize;
        pos += 4;
        if pos + len > blob.len() {
            break;
        }
        let bytes = blob[pos..pos + len].to_vec();
        pos += len;
        // SAFETY: bytes came from the kernel argv blob; valid OsString encoding.
        result.push(unsafe { OsString::from_encoded_bytes_unchecked(bytes) });
    }
    Args::new(result)
}
