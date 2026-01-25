use core::ffi::c_void;

use crate::ffi::CStr;
use crate::num::NonZero;
use crate::os::windows::io::{AsRawHandle, FromRawHandle, HandleOrNull};
use crate::sync::atomic::Ordering::Relaxed;
use crate::sync::atomic::{Atomic, AtomicBool};
use crate::sys::handle::Handle;
use crate::sys::pal::to_u16s;
use crate::sys::{FromInner, c, stack_overflow};
use crate::thread::ThreadInit;
use crate::time::Duration;
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
    // Preserve the zero duration behavior of `Sleep`.
    if dur.is_zero() {
        unsafe { c::Sleep(0) };
        return;
    }

    static HIGH_RESULTION_SUPPORTED: Atomic<bool> = AtomicBool::new(true);

    // Otherwise, create a waitable timer object in order to pass the system
    // a more accurate sleep time – waitable timer objects can be set with
    // 100 ns precision instead of the millisecond precision of `Sleep`.
    //
    // Incidentally, this also bypasses an issue with `Sleep`: `Sleep` can
    // return before the duration has elapsed since it is allowed to round
    // the sleep duration *down* to a clock interval (see #149935).
    let handle = loop {
        // Use high-precision sleep if available (Windows 10, version 1803+).
        let flags = if HIGH_RESULTION_SUPPORTED.load(Relaxed) {
            c::CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
        } else {
            0
        };
        let handle = unsafe {
            c::CreateWaitableTimerExW(ptr::null(), ptr::null(), flags, c::TIMER_ALL_ACCESS)
        };

        if !handle.is_null() {
            break unsafe { Handle::from_raw_handle(handle) };
        } else {
            match unsafe { c::GetLastError() } {
                c::ERROR_INVALID_PARAMETER if flags != 0 => {
                    HIGH_RESULTION_SUPPORTED.store(false, Relaxed);
                    continue;
                }
                error => {
                    panic!(
                        "failed to create waitable timer for sleep: {}",
                        io::Error::from_raw_os_error(error as i32)
                    );
                }
            }
        }
    };

    // Round up to sleep for at least the required duration.
    let mut intervals = dur.as_nanos().div_ceil(100);
    while intervals > 0 {
        // Set the timer. Since relative durations are negative, we can sleep
        // at most -i64::MIN intervals at a time.
        let to_sleep = u128::min(intervals, i64::MIN as u128);
        let time_param = (to_sleep as i64).wrapping_neg();
        let result = unsafe {
            c::SetWaitableTimer(handle.as_raw_handle(), &time_param, 0, None, ptr::null(), c::FALSE)
        };
        if result == 0 {
            panic!("failed to set waitable timer for sleep: {}", io::Error::last_os_error());
        }

        let result = unsafe { c::WaitForSingleObject(handle.as_raw_handle(), c::INFINITE) };
        if result != c::WAIT_OBJECT_0 {
            panic!("failed to wait on waitable timer for sleep: {}", io::Error::last_os_error());
        }

        intervals -= to_sleep;
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
