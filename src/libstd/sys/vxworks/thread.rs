use crate::cmp;
use crate::ffi::CStr;
use crate::io;
use crate::mem;
use crate::ptr;
use crate::sys::os;
use crate::time::Duration;

use crate::sys_common::thread::*;

pub const DEFAULT_MIN_STACK_SIZE: usize = 0x40000; // 256K

pub struct Thread {
    id: libc::pthread_t,
}

// Some platforms may have pthread_t as a pointer in which case we still want
// a thread to be Send/Sync
unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

// The pthread_attr_setstacksize symbol doesn't exist in the emscripten libc,
// so we have to not link to it to satisfy emcc's ERROR_ON_UNDEFINED_SYMBOLS.
unsafe fn pthread_attr_setstacksize(attr: *mut libc::pthread_attr_t,
                                    stack_size: libc::size_t) -> libc::c_int {
    libc::pthread_attr_setstacksize(attr, stack_size)
}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>)
                          -> io::Result<Thread> {
        let p = box p;
        let mut native: libc::pthread_t = mem::zeroed();
        let mut attr: libc::pthread_attr_t = mem::zeroed();
        assert_eq!(libc::pthread_attr_init(&mut attr), 0);

        let stack_size = cmp::max(stack, min_stack_size(&attr));

        match pthread_attr_setstacksize(&mut attr,
                                        stack_size) {
            0 => {}
            n => {
                assert_eq!(n, libc::EINVAL);
                // EINVAL means |stack_size| is either too small or not a
                // multiple of the system page size.  Because it's definitely
                // >= PTHREAD_STACK_MIN, it must be an alignment issue.
                // Round up to the nearest page and try again.
                let page_size = os::page_size();
                let stack_size = (stack_size + page_size - 1) &
                                 (-(page_size as isize - 1) as usize - 1);
                assert_eq!(libc::pthread_attr_setstacksize(&mut attr,
                                                           stack_size), 0);
            }
        };

        let ret = libc::pthread_create(&mut native, &attr, thread_start,
                                       &*p as *const _ as *mut _);
        assert_eq!(libc::pthread_attr_destroy(&mut attr), 0);

        return if ret != 0 {
            Err(io::Error::from_raw_os_error(ret))
        } else {
            mem::forget(p); // ownership passed to pthread_create
            Ok(Thread { id: native })
        };

        extern fn thread_start(main: *mut libc::c_void) -> *mut libc::c_void {
            unsafe { start_thread(main as *mut u8); }
            ptr::null_mut()
        }
    }

    pub fn yield_now() {
        let ret = unsafe { libc::sched_yield() };
        debug_assert_eq!(ret, 0);
    }

    pub fn set_name(_name: &CStr) {
        // VxWorks does not provide a way to set the task name except at creation time
    }

    pub fn sleep(dur: Duration) {
        let mut secs = dur.as_secs();
        let mut nsecs = dur.subsec_nanos() as _;

        // If we're awoken with a signal then the return value will be -1 and
        // nanosleep will fill in `ts` with the remaining time.
        unsafe {
            while secs > 0 || nsecs > 0 {
                let mut ts = libc::timespec {
                    tv_sec: cmp::min(libc::time_t::max_value() as u64, secs) as libc::time_t,
                    tv_nsec: nsecs,
                };
                secs -= ts.tv_sec as u64;
                if libc::nanosleep(&ts, &mut ts) == -1 {
                    assert_eq!(os::errno(), libc::EINTR);
                    secs += ts.tv_sec as u64;
                    nsecs = ts.tv_nsec;
                } else {
                    nsecs = 0;
                }
            }
        }
    }

    pub fn join(self) {
        unsafe {
            let ret = libc::pthread_join(self.id, ptr::null_mut());
            mem::forget(self);
            assert!(ret == 0,
                    "failed to join thread: {}", io::Error::from_raw_os_error(ret));
        }
    }

    pub fn id(&self) -> libc::pthread_t { self.id }

    pub fn into_id(self) -> libc::pthread_t {
        let id = self.id;
        mem::forget(self);
        id
    }
}

impl Drop for Thread {
    fn drop(&mut self) {
        let ret = unsafe { libc::pthread_detach(self.id) };
        debug_assert_eq!(ret, 0);
    }
}

#[cfg_attr(test, allow(dead_code))]
pub mod guard {
    use crate::ops::Range;
    pub type Guard = Range<usize>;
    pub unsafe fn current() -> Option<Guard> { None }
    pub unsafe fn init() -> Option<Guard> { None }
    pub unsafe fn deinit() {}
}

fn min_stack_size(_: *const libc::pthread_attr_t) -> usize {
    libc::PTHREAD_STACK_MIN
}
