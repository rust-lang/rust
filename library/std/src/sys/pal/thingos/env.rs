//! ThingOS environment-variable implementation.
//!
//! Wires `std::env::{var, vars, set_var, remove_var}` to kernel syscalls.
//!
//! **Note:** Environment storage is currently process-local only. Child
//! processes inherit a snapshot of the parent's environment at spawn time
//! (handled by the kernel), but there is no shared/global env store.
//!
//! # Syscalls used (abi/src/numbers.rs)
//! | Syscall        | Number | Purpose              |
//! |----------------|--------|----------------------|
//! | SYS_ENV_GET    | 0x1101 | Read one variable    |
//! | SYS_ENV_SET    | 0x1102 | Write one variable   |
//! | SYS_ENV_UNSET  | 0x1103 | Delete one variable  |
//! | SYS_ENV_LIST   | 0x1104 | List all variables   |

pub use super::common::Env;
use crate::ffi::{OsStr, OsString};

const SYS_ENV_GET: u32 = 0x1101;
const SYS_ENV_SET: u32 = 0x1102;
const SYS_ENV_UNSET: u32 = 0x1103;
const SYS_ENV_LIST: u32 = 0x1104;

#[inline(always)]
unsafe fn raw_syscall6(
    n: u32,
    a0: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
) -> isize {
    unsafe { crate::sys::pal::raw_syscall6(n, a0, a1, a2, a3, a4, a5) }
}

#[inline]
fn syscall_err(ret: isize) -> crate::io::Error {
    crate::io::Error::from_raw_os_error((-ret) as i32)
}

/// List all environment variables.
///
/// Blob format (kernel/src/syscall/handlers/process.rs `sys_env_list`):
///   count:   u32 LE
///   for each entry: key_len: u32 LE, key_bytes, val_len: u32 LE, val_bytes
pub fn env() -> Env {

    let needed = unsafe { raw_syscall6(SYS_ENV_LIST, 0, 0, 0, 0, 0, 0) };
    if needed <= 0 {
        return Env::new(crate::vec![]);
    }
    let mut buf = crate::vec![0u8; needed as usize];
    let ret =
        unsafe { raw_syscall6(SYS_ENV_LIST, buf.as_mut_ptr() as usize, buf.len(), 0, 0, 0, 0) };
    if ret < 0 {
        return Env::new(crate::vec![]);
    }

    let blob = &buf[..];
    if blob.len() < 4 {
        return Env::new(crate::vec![]);
    }
    let count = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
    let mut pos = 4usize;
    let mut pairs: crate::vec::Vec<(OsString, OsString)> = crate::vec::Vec::with_capacity(count);

    for _ in 0..count {
        if pos + 4 > blob.len() {
            break;
        }
        let key_len =
            u32::from_le_bytes([blob[pos], blob[pos + 1], blob[pos + 2], blob[pos + 3]]) as usize;
        pos += 4;
        if pos + key_len > blob.len() {
            break;
        }
        let key = unsafe { OsString::from_encoded_bytes_unchecked(blob[pos..pos + key_len].to_vec()) };
        pos += key_len;

        if pos + 4 > blob.len() {
            break;
        }
        let val_len =
            u32::from_le_bytes([blob[pos], blob[pos + 1], blob[pos + 2], blob[pos + 3]]) as usize;
        pos += 4;
        if pos + val_len > blob.len() {
            break;
        }
        let val = unsafe { OsString::from_encoded_bytes_unchecked(blob[pos..pos + val_len].to_vec()) };
        pos += val_len;

        pairs.push((key, val));
    }

    Env::new(pairs)
}

pub fn getenv(k: &OsStr) -> Option<OsString> {

    let kb = k.as_encoded_bytes();
    // Probe for needed length.
    let needed =
        unsafe { raw_syscall6(SYS_ENV_GET as u32, kb.as_ptr() as usize, kb.len(), 0, 0, 0, 0) };
    if needed < 0 {
        return None;
    }
    let mut val = crate::vec![0u8; needed as usize];
    let ret = unsafe {
        raw_syscall6(
            SYS_ENV_GET as u32,
            kb.as_ptr() as usize,
            kb.len(),
            val.as_mut_ptr() as usize,
            val.len(),
            0,
            0,
        )
    };
    if ret < 0 {
        return None;
    }
    Some(unsafe { OsString::from_encoded_bytes_unchecked(val) })
}

pub unsafe fn setenv(k: &OsStr, v: &OsStr) -> crate::io::Result<()> {
    let kb = k.as_encoded_bytes();
    let vb = v.as_encoded_bytes();
    let ret = unsafe {
        raw_syscall6(
            SYS_ENV_SET as u32,
            kb.as_ptr() as usize,
            kb.len(),
            vb.as_ptr() as usize,
            vb.len(),
            0,
            0,
        )
    };
    if ret < 0 { Err(syscall_err(ret)) } else { Ok(()) }
}

pub unsafe fn unsetenv(k: &OsStr) -> crate::io::Result<()> {
    let kb = k.as_encoded_bytes();
    let ret =
        unsafe { raw_syscall6(SYS_ENV_UNSET as u32, kb.as_ptr() as usize, kb.len(), 0, 0, 0, 0) };
    if ret < 0 { Err(syscall_err(ret)) } else { Ok(()) }
}
