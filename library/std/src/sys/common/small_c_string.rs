use crate::ffi::{CStr, CString};
use crate::mem::MaybeUninit;
use crate::path::Path;
use crate::slice;
use crate::{io, ptr};

// Make sure to stay under 4096 so the compiler doesn't insert a probe frame:
// https://docs.rs/compiler_builtins/latest/compiler_builtins/probestack/index.html
#[cfg(not(target_os = "espidf"))]
const MAX_STACK_ALLOCATION: usize = 384;
#[cfg(target_os = "espidf")]
const MAX_STACK_ALLOCATION: usize = 32;

const NUL_ERR: io::Error =
    io::const_io_error!(io::ErrorKind::InvalidInput, "file name contained an unexpected NUL byte");

#[inline]
pub fn run_path_with_cstr<T, F>(path: &Path, f: F) -> io::Result<T>
where
    F: FnOnce(&CStr) -> io::Result<T>,
{
    run_with_cstr(path.as_os_str().as_os_str_bytes(), f)
}

#[inline]
pub fn run_with_cstr<T, F>(bytes: &[u8], f: F) -> io::Result<T>
where
    F: FnOnce(&CStr) -> io::Result<T>,
{
    if bytes.len() >= MAX_STACK_ALLOCATION {
        return run_with_cstr_allocating(bytes, f);
    }

    let mut buf = MaybeUninit::<[u8; MAX_STACK_ALLOCATION]>::uninit();
    let buf_ptr = buf.as_mut_ptr() as *mut u8;

    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), buf_ptr, bytes.len());
        buf_ptr.add(bytes.len()).write(0);
    }

    match CStr::from_bytes_with_nul(unsafe { slice::from_raw_parts(buf_ptr, bytes.len() + 1) }) {
        Ok(s) => f(s),
        Err(_) => Err(NUL_ERR),
    }
}

#[cold]
#[inline(never)]
fn run_with_cstr_allocating<T, F>(bytes: &[u8], f: F) -> io::Result<T>
where
    F: FnOnce(&CStr) -> io::Result<T>,
{
    match CString::new(bytes) {
        Ok(s) => f(&s),
        Err(_) => Err(NUL_ERR),
    }
}
