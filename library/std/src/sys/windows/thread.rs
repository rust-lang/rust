use crate::ffi::CStr;
use crate::io;
use crate::num::NonZeroUsize;
use crate::os::windows::io::{AsRawHandle, FromRawHandle};
use crate::ptr;
use crate::sys::c;
use crate::sys::handle::Handle;
use crate::sys::stack_overflow;
use crate::time::Duration;

use libc::c_void;

use super::to_u16s;

pub const DEFAULT_MIN_STACK_SIZE: usize = 2 * 1024 * 1024;

pub struct Thread {
    handle: Handle,
}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        let p = Box::into_raw(box p);

        // FIXME On UNIX, we guard against stack sizes that are too small but
        // that's because pthreads enforces that stacks are at least
        // PTHREAD_STACK_MIN bytes big.  Windows has no such lower limit, it's
        // just that below a certain threshold you can't do anything useful.
        // That threshold is application and architecture-specific, however.
        // Round up to the next 64 kB because that's what the NT kernel does,
        // might as well make it explicit.
        let stack_size = (stack + 0xfffe) & (!0xfffe);
        let ret = c::CreateThread(
            ptr::null_mut(),
            stack_size,
            thread_start,
            p as *mut _,
            c::STACK_SIZE_PARAM_IS_A_RESERVATION,
            ptr::null_mut(),
        );

        return if ret as usize == 0 {
            // The thread failed to start and as a result p was not consumed. Therefore, it is
            // safe to reconstruct the box so that it gets deallocated.
            drop(Box::from_raw(p));
            Err(io::Error::last_os_error())
        } else {
            Ok(Thread { handle: Handle::from_raw_handle(ret) })
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
        unsafe { c::Sleep(super::dur2timeout(dur)) }
    }

    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    pub fn into_handle(self) -> Handle {
        self.handle
    }
}

pub fn available_concurrency() -> io::Result<NonZeroUsize> {
    let res = unsafe {
        let mut sysinfo: c::SYSTEM_INFO = crate::mem::zeroed();
        c::GetSystemInfo(&mut sysinfo);
        sysinfo.dwNumberOfProcessors as usize
    };
    match res {
        0 => Err(io::Error::new_const(
            io::ErrorKind::NotFound,
            &"The number of hardware threads is not known for the target platform",
        )),
        cpus => Ok(unsafe { NonZeroUsize::new_unchecked(cpus) }),
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
