use core::slice::memchr;

pub use super::common::Env;
use crate::ffi::{CStr, OsStr, OsString};
use crate::io;
use crate::os::wasi::prelude::*;
use crate::sys::common::small_c_string::run_with_cstr;
use crate::sys::pal::os::{cvt, libc};

cfg_select! {
    target_feature = "atomics" => {
        // Access to the environment must be protected by a lock in multi-threaded scenarios.
        use crate::sync::{PoisonError, RwLock};
        static ENV_LOCK: RwLock<()> = RwLock::new(());
        pub fn env_read_lock() -> impl Drop {
            ENV_LOCK.read().unwrap_or_else(PoisonError::into_inner)
        }
        pub fn env_write_lock() -> impl Drop {
            ENV_LOCK.write().unwrap_or_else(PoisonError::into_inner)
        }
    }
    _ => {
        // No need for a lock if we are single-threaded.
        pub fn env_read_lock() -> impl Drop {
            Box::new(())
        }
        pub fn env_write_lock() -> impl Drop {
            Box::new(())
        }
    }
}

pub fn env() -> Env {
    unsafe {
        let _guard = env_read_lock();

        // Use `__wasilibc_get_environ` instead of `environ` here so that we
        // don't require wasi-libc to eagerly initialize the environment
        // variables.
        let mut environ = libc::__wasilibc_get_environ();

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

    // See src/libstd/sys/pal/unix/os.rs, same as that
    fn parse(input: &[u8]) -> Option<(OsString, OsString)> {
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
        run_with_cstr(v.as_bytes(), &|v| unsafe {
            let _guard = env_write_lock();
            cvt(libc::setenv(k.as_ptr(), v.as_ptr(), 1)).map(drop)
        })
    })
}

pub unsafe fn unsetenv(n: &OsStr) -> io::Result<()> {
    run_with_cstr(n.as_bytes(), &|nbuf| unsafe {
        let _guard = env_write_lock();
        cvt(libc::unsetenv(nbuf.as_ptr())).map(drop)
    })
}
