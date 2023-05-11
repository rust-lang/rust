//! Bindings to acquire a global named lock.
//!
//! This is intended to be used to synchronize multiple compiler processes to
//! ensure that we can output complete errors without interleaving on Windows.
//! Note that this is currently only needed for allowing only one 32-bit MSVC
//! linker to execute at once on MSVC hosts, so this is only implemented for
//! `cfg(windows)`. Also note that this may not always be used on Windows,
//! only when targeting 32-bit MSVC.
//!
//! For more information about why this is necessary, see where this is called.

use std::any::Any;

#[cfg(windows)]
pub fn acquire_global_lock(name: &str) -> Box<dyn Any> {
    use std::ffi::CString;
    use std::io;

    use windows::{
        core::PCSTR,
        Win32::Foundation::{CloseHandle, HANDLE, WAIT_ABANDONED, WAIT_OBJECT_0},
        Win32::System::Threading::{CreateMutexA, ReleaseMutex, WaitForSingleObject},
        Win32::System::WindowsProgramming::INFINITE,
    };

    struct Handle(HANDLE);

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }

    struct Guard(Handle);

    impl Drop for Guard {
        fn drop(&mut self) {
            unsafe {
                ReleaseMutex((self.0).0);
            }
        }
    }

    let cname = CString::new(name).unwrap();
    // Create a named mutex, with no security attributes and also not
    // acquired when we create it.
    //
    // This will silently create one if it doesn't already exist, or it'll
    // open up a handle to one if it already exists.
    let mutex = unsafe { CreateMutexA(None, false, PCSTR::from_raw(cname.as_ptr().cast())) }
        .unwrap_or_else(|_| panic!("failed to create global mutex named `{}`", name));
    let mutex = Handle(mutex);

    // Acquire the lock through `WaitForSingleObject`.
    //
    // A return value of `WAIT_OBJECT_0` means we successfully acquired it.
    //
    // A return value of `WAIT_ABANDONED` means that the previous holder of
    // the thread exited without calling `ReleaseMutex`. This can happen,
    // for example, when the compiler crashes or is interrupted via ctrl-c
    // or the like. In this case, however, we are still transferred
    // ownership of the lock so we continue.
    //
    // If an error happens.. well... that's surprising!
    match unsafe { WaitForSingleObject(mutex.0, INFINITE) } {
        WAIT_OBJECT_0 | WAIT_ABANDONED => (),
        err => panic!(
            "WaitForSingleObject failed on global mutex named `{}`: {} (ret={:x})",
            name,
            io::Error::last_os_error(),
            err.0
        ),
    }

    // Return a guard which will call `ReleaseMutex` when dropped.
    Box::new(Guard(mutex))
}

#[cfg(not(windows))]
pub fn acquire_global_lock(_name: &str) -> Box<dyn Any> {
    Box::new(())
}
