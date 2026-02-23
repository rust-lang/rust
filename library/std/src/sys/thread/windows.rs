use core::ffi::c_void;

use crate::ffi::CStr;
use crate::num::NonZero;
use crate::os::windows::io::{AsRawHandle, HandleOrNull};
use crate::sys::handle::Handle;
use crate::sys::pal::time::WaitableTimer;
use crate::sys::pal::{dur2timeout, to_u16s};
use crate::sys::{FromInner, c, stack_overflow};
use crate::thread::ThreadInit;
use crate::time::{Duration, Instant};
use crate::{io, ptr};

pub const DEFAULT_MIN_STACK_SIZE: usize = 2 * 1024 * 1024;

pub struct Thread {
    handle: Handle,
}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub unsafe fn new(stack: usize, init: Box<ThreadInit>) -> io::Result<Thread> {
        let data = Box::into_raw(init);

        // CreateThread rounds up values for the stack size to the nearest page size (at least 4kb).
        // If a value of zero is given then the default stack size is used instead.
        // SAFETY: `thread_start` has the right ABI for a thread's entry point.
        // `data` is simply passed through to the new thread without being touched.
        let ret = unsafe {
            let ret = c::CreateThread(
                ptr::null_mut(),
                stack,
                Some(thread_start),
                data as *mut _,
                c::STACK_SIZE_PARAM_IS_A_RESERVATION,
                ptr::null_mut(),
            );
            HandleOrNull::from_raw_handle(ret)
        };
        return if let Ok(handle) = ret.try_into() {
            Ok(Thread { handle: Handle::from_inner(handle) })
        } else {
            // The thread failed to start and as a result data was not consumed. Therefore, it is
            // safe to reconstruct the box so that it gets deallocated.
            unsafe { drop(Box::from_raw(data)) };
            Err(io::Error::last_os_error())
        };

        unsafe extern "system" fn thread_start(data: *mut c_void) -> u32 {
            // SAFETY: we are simply recreating the box that was leaked earlier.
            let init = unsafe { Box::from_raw(data as *mut ThreadInit) };
            let rust_start = init.init();

            // Reserve some stack space for if we otherwise run out of stack.
            stack_overflow::reserve_stack();

            rust_start();
            0
        }
    }

    pub fn join(self) {
        let rc = unsafe { c::WaitForSingleObject(self.handle.as_raw_handle(), c::INFINITE) };
        if rc == c::WAIT_FAILED {
            panic!("failed to join on thread: {}", io::Error::last_os_error());
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

pub fn current_os_id() -> Option<u64> {
    // SAFETY: FFI call with no preconditions.
    let id: u32 = unsafe { c::GetCurrentThreadId() };

    // A return value of 0 indicates failed lookup.
    if id == 0 { None } else { Some(id.into()) }
}

pub fn set_name(name: &CStr) {
    if let Ok(utf8) = name.to_str() {
        if let Ok(utf16) = to_u16s(utf8) {
            unsafe {
                // SAFETY: the vec returned by `to_u16s` ends with a zero value
                set_name_wide(&utf16)
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

pub fn sleep(dur: Duration) {
    fn high_precision_sleep(dur: Duration) -> Result<(), ()> {
        let timer = WaitableTimer::high_resolution()?;
        timer.set(dur)?;
        timer.wait()
    }
    // Directly forward to `Sleep` for its zero duration behavior when indeed
    // zero in order to skip the `Instant::now` calls, useless in this case.
    if dur.is_zero() {
        unsafe { c::Sleep(0) };
    // Attempt to use high-precision sleep (Windows 10, version 1803+).
    // On error, fallback to the standard `Sleep` function.
    } else if high_precision_sleep(dur).is_err() {
        let start = Instant::now();
        unsafe { c::Sleep(dur2timeout(dur)) };

        // See #149935: `Sleep` under Windows 7 and probably 8 as well seems a
        // bit buggy for us as it can last less than the requested time while
        // our API is meant to guarantee that. This is fixed by measuring the
        // effective time difference and if needed, sleeping a bit more in
        // order to ensure the duration is always exceeded. A fixed single
        // millisecond works because `Sleep` operates based on a system-wide
        // (until Windows 10 2004 that makes it process-local) interrupt timer
        // that counts in "tick" units of ~15ms by default: a 1ms timeout
        // therefore passes the next tick boundary.
        if start.elapsed() < dur {
            unsafe { c::Sleep(1) };
        }
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
