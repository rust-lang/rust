use crate::ffi::CStr;
use crate::mem::{self, ManuallyDrop};
use crate::num::NonZero;
use crate::sys::os;
use crate::time::Duration;
use crate::{cmp, io, ptr};

pub const DEFAULT_MIN_STACK_SIZE: usize = 8 * 1024;

pub struct Thread {
    id: libc::pthread_t,
}

// Some platforms may have pthread_t as a pointer in which case we still want
// a thread to be Send/Sync
unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

unsafe extern "C" {
    pub fn TEE_Wait(timeout: u32) -> u32;
}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        let p = Box::into_raw(Box::new(p));
        let mut native: libc::pthread_t = unsafe { mem::zeroed() };
        let mut attr: libc::pthread_attr_t = unsafe { mem::zeroed() };
        assert_eq!(unsafe { libc::pthread_attr_init(&mut attr) }, 0);
        assert_eq!(
            unsafe {
                libc::pthread_attr_settee(
                    &mut attr,
                    libc::TEESMP_THREAD_ATTR_CA_INHERIT,
                    libc::TEESMP_THREAD_ATTR_TASK_ID_INHERIT,
                    libc::TEESMP_THREAD_ATTR_HAS_SHADOW,
                )
            },
            0,
        );

        let stack_size = cmp::max(stack, min_stack_size(&attr));

        match unsafe { libc::pthread_attr_setstacksize(&mut attr, stack_size) } {
            0 => {}
            n => {
                assert_eq!(n, libc::EINVAL);
                // EINVAL means |stack_size| is either too small or not a
                // multiple of the system page size.  Because it's definitely
                // >= PTHREAD_STACK_MIN, it must be an alignment issue.
                // Round up to the nearest page and try again.
                let page_size = os::page_size();
                let stack_size =
                    (stack_size + page_size - 1) & (-(page_size as isize - 1) as usize - 1);
                assert_eq!(unsafe { libc::pthread_attr_setstacksize(&mut attr, stack_size) }, 0);
            }
        };

        let ret = libc::pthread_create(&mut native, &attr, thread_start, p as *mut _);
        // Note: if the thread creation fails and this assert fails, then p will
        // be leaked. However, an alternative design could cause double-free
        // which is clearly worse.
        assert_eq!(unsafe { libc::pthread_attr_destroy(&mut attr) }, 0);

        return if ret != 0 {
            // The thread failed to start and as a result p was not consumed. Therefore, it is
            // safe to reconstruct the box so that it gets deallocated.
            drop(unsafe { Box::from_raw(p) });
            Err(io::Error::from_raw_os_error(ret))
        } else {
            // The new thread will start running earliest after the next yield.
            // We add a yield here, so that the user does not have to.
            Thread::yield_now();
            Ok(Thread { id: native })
        };

        extern "C" fn thread_start(main: *mut libc::c_void) -> *mut libc::c_void {
            unsafe {
                // Next, set up our stack overflow handler which may get triggered if we run
                // out of stack.
                // this is not necessary in TEE.
                //let _handler = stack_overflow::Handler::new();
                // Finally, let's run some code.
                Box::from_raw(main as *mut Box<dyn FnOnce()>)();
            }
            ptr::null_mut()
        }
    }

    pub fn yield_now() {
        let ret = unsafe { libc::sched_yield() };
        debug_assert_eq!(ret, 0);
    }

    /// This does not do anything on teeos
    pub fn set_name(_name: &CStr) {
        // Both pthread_setname_np and prctl are not available to the TA,
        // so we can't implement this currently. If the need arises please
        // contact the teeos rustzone team.
    }

    /// only main thread could wait for sometime in teeos
    pub fn sleep(dur: Duration) {
        let sleep_millis = dur.as_millis();
        let final_sleep: u32 =
            if sleep_millis >= u32::MAX as u128 { u32::MAX } else { sleep_millis as u32 };
        unsafe {
            let _ = TEE_Wait(final_sleep);
        }
    }

    /// must join, because no pthread_detach supported
    pub fn join(self) {
        let id = self.into_id();
        let ret = unsafe { libc::pthread_join(id, ptr::null_mut()) };
        assert!(ret == 0, "failed to join thread: {}", io::Error::from_raw_os_error(ret));
    }

    pub fn id(&self) -> libc::pthread_t {
        self.id
    }

    pub fn into_id(self) -> libc::pthread_t {
        ManuallyDrop::new(self).id
    }
}

impl Drop for Thread {
    fn drop(&mut self) {
        // we can not call detach, so just panic if thread spawn without join
        panic!("thread must join, detach is not supported!");
    }
}

// Note: Both `sched_getaffinity` and `sysconf` are available but not functional on
// teeos, so this function always returns an Error!
pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    Err(io::Error::UNKNOWN_THREAD_COUNT)
}

fn min_stack_size(_: *const libc::pthread_attr_t) -> usize {
    libc::PTHREAD_STACK_MIN.try_into().expect("Infallible")
}
