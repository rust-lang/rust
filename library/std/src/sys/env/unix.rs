use core::slice::memchr;

use libc::c_char;

pub use super::common::Env;
use crate::ffi::{CStr, OsStr, OsString};
use crate::io;
use crate::os::unix::prelude::*;
use crate::sync::{PoisonError, RwLock};
use crate::sys::common::small_c_string::run_with_cstr;
use crate::sys::cvt;

// Use `_NSGetEnviron` on Apple platforms.
//
// `_NSGetEnviron` is the documented alternative (see `man environ`), and has
// been available since the first versions of both macOS and iOS.
//
// Nowadays, specifically since macOS 10.8, `environ` has been exposed through
// `libdyld.dylib`, which is linked via. `libSystem.dylib`:
// <https://github.com/apple-oss-distributions/dyld/blob/dyld-1160.6/libdyld/libdyldGlue.cpp#L913>
//
// So in the end, it likely doesn't really matter which option we use, but the
// performance cost of using `_NSGetEnviron` is extremely miniscule, and it
// might be ever so slightly more supported, so let's just use that.
//
// NOTE: The header where this is defined (`crt_externs.h`) was added to the
// iOS 13.0 SDK, which has been the source of a great deal of confusion in the
// past about the availability of this API.
//
// NOTE(madsmtm): Neither this nor using `environ` has been verified to not
// cause App Store rejections; if this is found to be the case, an alternative
// implementation of this is possible using `[NSProcessInfo environment]`
// - which internally uses `_NSGetEnviron` and a system-wide lock on the
// environment variables to protect against `setenv`, so using that might be
// desirable anyhow? Though it also means that we have to link to Foundation.
#[cfg(target_vendor = "apple")]
pub unsafe fn environ() -> *mut *const *const c_char {
    unsafe { libc::_NSGetEnviron() as *mut *const *const c_char }
}

// Use the `environ` static which is part of POSIX.
#[cfg(not(target_vendor = "apple"))]
pub unsafe fn environ() -> *mut *const *const c_char {
    unsafe extern "C" {
        static mut environ: *const *const c_char;
    }
    &raw mut environ
}

static ENV_LOCK: RwLock<()> = RwLock::new(());

pub fn env_read_lock() -> impl Drop {
    ENV_LOCK.read().unwrap_or_else(PoisonError::into_inner)
}

/// Returns a vector of (variable, value) byte-vector pairs for all the
/// environment variables of the current process.
pub fn env() -> Env {
    unsafe {
        let _guard = env_read_lock();
        let mut environ = *environ();
        let mut result = Vec::new();
        if !environ.is_null() {
            while !(*environ).is_null() {
                if let Some(key_value) = parse(CStr::from_ptr(*environ).to_bytes()) {
                    result.push(key_value);
                }
                environ = environ.add(1);
            }
        }
        return Env::new(result);
    }

    fn parse(input: &[u8]) -> Option<(OsString, OsString)> {
        // Strategy (copied from glibc): Variable name and value are separated
        // by an ASCII equals sign '='. Since a variable name must not be
        // empty, allow variable names starting with an equals sign. Skip all
        // malformed lines.
        if input.is_empty() {
            return None;
        }
        let pos = memchr::memchr(b'=', &input[1..]).map(|p| p + 1);
        pos.map(|p| {
            (
                OsStringExt::from_vec(input[..p].to_vec()),
                OsStringExt::from_vec(input[p + 1..].to_vec()),
            )
        })
    }
}

pub fn getenv(k: &OsStr) -> Option<OsString> {
    // environment variables with a nul byte can't be set, so their value is
    // always None as well
    run_with_cstr(k.as_bytes(), &|k| {
        let _guard = env_read_lock();
        let v = unsafe { libc::getenv(k.as_ptr()) } as *const libc::c_char;

        if v.is_null() {
            Ok(None)
        } else {
            // SAFETY: `v` cannot be mutated while executing this line since we've a read lock
            let bytes = unsafe { CStr::from_ptr(v) }.to_bytes().to_vec();

            Ok(Some(OsStringExt::from_vec(bytes)))
        }
    })
    .ok()
    .flatten()
}

pub unsafe fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    run_with_cstr(k.as_bytes(), &|k| {
        run_with_cstr(v.as_bytes(), &|v| {
            let _guard = ENV_LOCK.write();
            cvt(unsafe { libc::setenv(k.as_ptr(), v.as_ptr(), 1) }).map(drop)
        })
    })
}

pub unsafe fn unsetenv(n: &OsStr) -> io::Result<()> {
    run_with_cstr(n.as_bytes(), &|nbuf| {
        let _guard = ENV_LOCK.write();
        cvt(unsafe { libc::unsetenv(nbuf.as_ptr()) }).map(drop)
    })
}
