use core::slice::memchr;

pub use super::common::Env;
use crate::ffi::{CStr, OsStr, OsString};
use crate::io;
use crate::os::raw::{c_char, c_int};
use crate::os::solid::ffi::{OsStrExt, OsStringExt};
use crate::sync::{PoisonError, RwLock};
use crate::sys::common::small_c_string::run_with_cstr;

static ENV_LOCK: RwLock<()> = RwLock::new(());

pub fn env_read_lock() -> impl Drop {
    ENV_LOCK.read().unwrap_or_else(PoisonError::into_inner)
}

/// Returns a vector of (variable, value) byte-vector pairs for all the
/// environment variables of the current process.
pub fn env() -> Env {
    unsafe extern "C" {
        static mut environ: *const *const c_char;
    }

    unsafe {
        let _guard = env_read_lock();
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
            cvt_env(unsafe { libc::setenv(k.as_ptr(), v.as_ptr(), 1) }).map(drop)
        })
    })
}

pub unsafe fn unsetenv(n: &OsStr) -> io::Result<()> {
    run_with_cstr(n.as_bytes(), &|nbuf| {
        let _guard = ENV_LOCK.write();
        cvt_env(unsafe { libc::unsetenv(nbuf.as_ptr()) }).map(drop)
    })
}

/// In kmclib, `setenv` and `unsetenv` don't always set `errno`, so this
/// function just returns a generic error.
fn cvt_env(t: c_int) -> io::Result<c_int> {
    if t == -1 { Err(io::const_error!(io::ErrorKind::Uncategorized, "failure")) } else { Ok(t) }
}
