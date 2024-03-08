use crate::ffi::CStr;
use crate::io;
use crate::num::NonZero;
use crate::os::windows::io::AsRawHandle;
use crate::os::windows::io::HandleOrNull;
use crate::ptr;
use crate::sys::c;
use crate::sys::handle::Handle;
use crate::sys::stack_overflow;
use crate::sys_common::FromInner;
use crate::time::Duration;
use alloc::ffi::CString;
use core::ffi::c_void;

use super::time::WaitableTimer;
use super::to_u16s;

pub const DEFAULT_MIN_STACK_SIZE: usize = 2 * 1024 * 1024;

pub struct Thread {
    handle: Handle,
}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        let p = Box::into_raw(Box::new(p));

        // CreateThread rounds up values for the stack size to the nearest page size (at least 4kb).
        // If a value of zero is given then the default stack size is used instead.
        let ret = c::CreateThread(
            ptr::null_mut(),
            stack,
            Some(thread_start),
            p as *mut _,
            c::STACK_SIZE_PARAM_IS_A_RESERVATION,
            ptr::null_mut(),
        );
        let ret = HandleOrNull::from_raw_handle(ret);
        return if let Ok(handle) = ret.try_into() {
            Ok(Thread { handle: Handle::from_inner(handle) })
        } else {
            // The thread failed to start and as a result p was not consumed. Therefore, it is
            // safe to reconstruct the box so that it gets deallocated.
            drop(Box::from_raw(p));
            Err(io::Error::last_os_error())
        };

        extern "system" fn thread_start(main: *mut c_void) -> c::DWORD {
            unsafe {
                // Next, set up our stack overflow handler which may get triggered if we run
                // out of stack.
                let _handler = stack_overflow::Handler::new();
                // Finally, let's run some code.
                Box::from_raw(main as *mut Box<dyn FnOnce()>)();
            }
            0
        }
    }

    pub fn set_name(name: &CStr) {
        if let Ok(utf8) = name.to_str() {
            if let Ok(utf16) = to_u16s(utf8) {
                unsafe {
                    c::SetThreadDescription(c::GetCurrentThread(), utf16.as_ptr());
                };
            };
        };
    }

    pub fn get_name() -> Option<CString> {
        unsafe {
            let mut ptr = core::ptr::null_mut();
            let result = c::GetThreadDescription(c::GetCurrentThread(), &mut ptr);
            if result < 0 {
                return None;
            }
            let name = String::from_utf16_lossy({
                let mut len = 0;
                while *ptr.add(len) != 0 {
                    len += 1;
                }
                core::slice::from_raw_parts(ptr, len)
            })
            .into_bytes();
            // Attempt to free the memory.
            // This should never fail but if it does then there's not much we can do about it.
            let result = c::LocalFree(ptr.cast::<c_void>());
            debug_assert!(result.is_null());
            if name.is_empty() { None } else { Some(CString::from_vec_unchecked(name)) }
        }
    }

    pub fn join(self) {
        let rc = unsafe { c::WaitForSingleObject(self.handle.as_raw_handle(), c::INFINITE) };
        if rc == c::WAIT_FAILED {
            panic!("failed to join on thread: {}", io::Error::last_os_error());
        }
    }

    pub fn yield_now() {
        // This function will return 0 if there are no other threads to execute,
        // but this also means that the yield was useless so this isn't really a
        // case that needs to be worried about.
        unsafe {
            c::SwitchToThread();
        }
    }

    pub fn sleep(dur: Duration) {
        fn high_precision_sleep(dur: Duration) -> Result<(), ()> {
            let timer = WaitableTimer::high_resolution()?;
            timer.set(dur)?;
            timer.wait()
        }
        // Attempt to use high-precision sleep (Windows 10, version 1803+).
        // On error fallback to the standard `Sleep` function.
        // Also preserves the zero duration behaviour of `Sleep`.
        if dur.is_zero() || high_precision_sleep(dur).is_err() {
            unsafe { c::Sleep(super::dur2timeout(dur)) }
        }
    }

    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    pub fn into_handle(self) -> Handle {
        self.handle
    }
}

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    let res = unsafe {
        let mut sysinfo: c::SYSTEM_INFO = crate::mem::zeroed();
        c::GetSystemInfo(&mut sysinfo);
        sysinfo.dwNumberOfProcessors as usize
    };
    match res {
        0 => Err(io::const_io_error!(
            io::ErrorKind::NotFound,
            "The number of hardware threads is not known for the target platform",
        )),
        cpus => Ok(unsafe { NonZero::new_unchecked(cpus) }),
    }
}

#[cfg_attr(test, allow(dead_code))]
pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> {
        None
    }
    pub unsafe fn init() -> Option<Guard> {
        None
    }
}
