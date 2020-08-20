//! Dynamic library facilities.
//!
//! A simple wrapper over the platform's dynamic library facilities

use std::ffi::CString;
use std::path::Path;

pub struct DynamicLibrary {
    handle: *mut u8,
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        unsafe { dl::close(self.handle) }
    }
}

impl DynamicLibrary {
    /// Lazily open a dynamic library.
    pub fn open(filename: &Path) -> Result<DynamicLibrary, String> {
        let maybe_library = dl::open(filename.as_os_str());

        // The dynamic library must not be constructed if there is
        // an error opening the library so the destructor does not
        // run.
        match maybe_library {
            Err(err) => Err(err),
            Ok(handle) => Ok(DynamicLibrary { handle }),
        }
    }

    /// Accesses the value at the symbol of the dynamic library.
    pub unsafe fn symbol<T>(&self, symbol: &str) -> Result<*mut T, String> {
        // This function should have a lifetime constraint of 'a on
        // T but that feature is still unimplemented

        let raw_string = CString::new(symbol).unwrap();
        let maybe_symbol_value = dl::symbol(self.handle, raw_string.as_ptr());

        // The value must not be constructed if there is an error so
        // the destructor does not run.
        match maybe_symbol_value {
            Err(err) => Err(err),
            Ok(symbol_value) => Ok(symbol_value as *mut T),
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(unix)]
mod dl {
    use std::ffi::{CString, OsStr};
    use std::os::unix::prelude::*;

    // `dlerror` is process global, so we can only allow a single thread at a
    // time to call `dlsym` and `dlopen` if we want to check the error message.
    mod error {
        use std::ffi::CStr;
        use std::lazy::SyncLazy;
        use std::sync::{Mutex, MutexGuard};

        pub fn lock() -> MutexGuard<'static, Guard> {
            static LOCK: SyncLazy<Mutex<Guard>> = SyncLazy::new(|| Mutex::new(Guard { _priv: () }));
            LOCK.lock().unwrap()
        }

        pub struct Guard {
            _priv: (),
        }

        impl Guard {
            pub fn get(&mut self) -> Result<(), String> {
                let msg = unsafe { libc::dlerror() };
                if msg.is_null() {
                    Ok(())
                } else {
                    let msg = unsafe { CStr::from_ptr(msg as *const _) };
                    Err(msg.to_string_lossy().into_owned())
                }
            }

            pub fn clear(&mut self) {
                let _ = unsafe { libc::dlerror() };
            }
        }
    }

    pub(super) fn open(filename: &OsStr) -> Result<*mut u8, String> {
        let s = CString::new(filename.as_bytes()).unwrap();

        let mut dlerror = error::lock();
        let ret = unsafe { libc::dlopen(s.as_ptr(), libc::RTLD_LAZY) } as *mut u8;

        if !ret.is_null() {
            return Ok(ret);
        }

        // A NULL return from `dlopen` indicates that an error has
        // definitely occurred, so if nothing is in `dlerror`, we are
        // racing with another thread that has stolen our error message.
        dlerror.get().and_then(|()| Err("Unknown error".to_string()))
    }

    pub(super) unsafe fn symbol(
        handle: *mut u8,
        symbol: *const libc::c_char,
    ) -> Result<*mut u8, String> {
        let mut dlerror = error::lock();

        // Flush `dlerror` since we need to use it to determine whether the subsequent call to
        // `dlsym` is successful.
        dlerror.clear();

        let ret = libc::dlsym(handle as *mut libc::c_void, symbol) as *mut u8;

        // A non-NULL return value *always* indicates success. There's no need
        // to check `dlerror`.
        if !ret.is_null() {
            return Ok(ret);
        }

        dlerror.get().map(|()| ret)
    }

    pub(super) unsafe fn close(handle: *mut u8) {
        libc::dlclose(handle as *mut libc::c_void);
    }
}

#[cfg(windows)]
mod dl {
    use std::ffi::OsStr;
    use std::io;
    use std::os::windows::prelude::*;
    use std::ptr;

    use winapi::shared::minwindef::HMODULE;
    use winapi::um::errhandlingapi::SetThreadErrorMode;
    use winapi::um::libloaderapi::{FreeLibrary, GetProcAddress, LoadLibraryW};
    use winapi::um::winbase::SEM_FAILCRITICALERRORS;

    pub(super) fn open(filename: &OsStr) -> Result<*mut u8, String> {
        // disable "dll load failed" error dialog.
        let prev_error_mode = unsafe {
            let new_error_mode = SEM_FAILCRITICALERRORS;
            let mut prev_error_mode = 0;
            let result = SetThreadErrorMode(new_error_mode, &mut prev_error_mode);
            if result == 0 {
                return Err(io::Error::last_os_error().to_string());
            }
            prev_error_mode
        };

        let filename_str: Vec<_> = filename.encode_wide().chain(Some(0)).collect();
        let result = unsafe { LoadLibraryW(filename_str.as_ptr()) } as *mut u8;
        let result = ptr_result(result);

        unsafe {
            SetThreadErrorMode(prev_error_mode, ptr::null_mut());
        }

        result
    }

    pub(super) unsafe fn symbol(
        handle: *mut u8,
        symbol: *const libc::c_char,
    ) -> Result<*mut u8, String> {
        let ptr = GetProcAddress(handle as HMODULE, symbol) as *mut u8;
        ptr_result(ptr)
    }

    pub(super) unsafe fn close(handle: *mut u8) {
        FreeLibrary(handle as HMODULE);
    }

    fn ptr_result<T>(ptr: *mut T) -> Result<*mut T, String> {
        if ptr.is_null() { Err(io::Error::last_os_error().to_string()) } else { Ok(ptr) }
    }
}
