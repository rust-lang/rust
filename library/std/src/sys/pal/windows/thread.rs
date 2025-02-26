use core::ffi::c_void;

use super::time::WaitableTimer;
use super::to_u16s;
use crate::ffi::CStr;
use crate::num::NonZero;
use crate::os::windows::io::{AsRawHandle, HandleOrNull};
use crate::sys::handle::Handle;
use crate::sys::{c, stack_overflow};
use crate::sys_common::FromInner;
use crate::time::Duration;
use crate::{io, ptr};

pub const DEFAULT_MIN_STACK_SIZE: usize = 2 * 1024 * 1024;

pub struct Thread {
    handle: Handle,
}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        let p = Box::into_raw(Box::new(p));

        // CreateThread rounds up values for the stack size to the nearest page size (at least 4kb).
        // If a value of zero is given then the default stack size is used instead.
        // SAFETY: `thread_start` has the right ABI for a thread's entry point.
        // `p` is simply passed through to the new thread without being touched.
        let ret = unsafe {
            let ret = c::CreateThread(
                ptr::null_mut(),
                stack,
                Some(thread_start),
                p as *mut _,
                c::STACK_SIZE_PARAM_IS_A_RESERVATION,
                ptr::null_mut(),
            );
            HandleOrNull::from_raw_handle(ret)
        };
        return if let Ok(handle) = ret.try_into() {
            Ok(Thread { handle: Handle::from_inner(handle) })
        } else {
            // The thread failed to start and as a result p was not consumed. Therefore, it is
            // safe to reconstruct the box so that it gets deallocated.
            unsafe { drop(Box::from_raw(p)) };
            Err(io::Error::last_os_error())
        };

        unsafe extern "system" fn thread_start(main: *mut c_void) -> u32 {
            // Next, reserve some stack space for if we otherwise run out of stack.
            stack_overflow::reserve_stack();
            // Finally, let's run some code.
            // SAFETY: We are simply recreating the box that was leaked earlier.
            // It's the responsibility of the one who call `Thread::new` to ensure this is safe to call here.
            unsafe { Box::from_raw(main as *mut Box<dyn FnOnce()>)() };
            0
        }
    }

    pub fn set_name(name: &CStr) {
        if let Ok(utf8) = name.to_str() {
            if let Ok(utf16) = to_u16s(utf8) {
                unsafe {
                    // SAFETY: the vec returned by `to_u16s` ends with a zero value
                    Self::set_name_wide(&utf16)
                }
            };
        };
    }

    /// # Safety
    ///
    /// `name` must end with a zero value
    pub unsafe fn set_name_wide(name: &[u16]) {
        unsafe { c::SetThreadDescription(c::GetCurrentThread(), name.as_ptr()) };
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
        // Also preserves the zero duration behavior of `Sleep`.
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
        0 => Err(io::Error::UNKNOWN_THREAD_COUNT),
        cpus => Ok(unsafe { NonZero::new_unchecked(cpus) }),
    }
}
