#![forbid(unsafe_op_in_unsafe_fn)]

use crate::ffi::{CStr, OsString};
use crate::io;
use crate::os::wasi::prelude::*;
use crate::path::{self, PathBuf};
use crate::sys::helpers::run_path_with_cstr;

pub fn getcwd() -> io::Result<PathBuf> {
    let mut buf = Vec::with_capacity(512);
    loop {
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut libc::c_char;
            if !libc::getcwd(ptr, buf.capacity()).is_null() {
                let len = CStr::from_ptr(buf.as_ptr() as *const libc::c_char).to_bytes().len();
                buf.set_len(len);
                buf.shrink_to_fit();
                return Ok(PathBuf::from(OsString::from_vec(buf)));
            } else {
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ERANGE) {
                    return Err(error);
                }
            }

            // Trigger the internal buffer resizing logic of `Vec` by requiring
            // more space than the current capacity.
            let cap = buf.capacity();
            buf.set_len(cap);
            buf.reserve(1);
        }
    }
}

pub fn chdir(p: &path::Path) -> io::Result<()> {
    let result = run_path_with_cstr(p, &|p| unsafe { Ok(libc::chdir(p.as_ptr())) })?;
    match result == (0 as libc::c_int) {
        true => Ok(()),
        false => Err(io::Error::last_os_error()),
    }
}

pub fn temp_dir() -> PathBuf {
    panic!("not supported by WASI yet")
}
