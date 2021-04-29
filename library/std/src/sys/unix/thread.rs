use crate::cmp;
use crate::ffi::CStr;
use crate::io;
use crate::mem;
use crate::num::NonZeroUsize;
use crate::ptr;
use crate::sys::{os, stack_overflow};
use crate::time::Duration;

#[cfg(not(any(target_os = "l4re", target_os = "vxworks")))]
pub const DEFAULT_MIN_STACK_SIZE: usize = 2 * 1024 * 1024;
#[cfg(target_os = "l4re")]
pub const DEFAULT_MIN_STACK_SIZE: usize = 1024 * 1024;
#[cfg(target_os = "vxworks")]
pub const DEFAULT_MIN_STACK_SIZE: usize = 256 * 1024;

pub struct Thread {
    id: libc::pthread_t,
}

// Some platforms may have pthread_t as a pointer in which case we still want
// a thread to be Send/Sync
unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        let p = Box::into_raw(box p);
        let mut native: libc::pthread_t = mem::zeroed();
        let mut attr: libc::pthread_attr_t = mem::zeroed();
        assert_eq!(libc::pthread_attr_init(&mut attr), 0);

        let stack_size = cmp::max(stack, min_stack_size(&attr));

        match libc::pthread_attr_setstacksize(&mut attr, stack_size) {
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
                assert_eq!(libc::pthread_attr_setstacksize(&mut attr, stack_size), 0);
            }
        };

        let ret = libc::pthread_create(&mut native, &attr, thread_start, p as *mut _);
        // Note: if the thread creation fails and this assert fails, then p will
        // be leaked. However, an alternative design could cause double-free
        // which is clearly worse.
        assert_eq!(libc::pthread_attr_destroy(&mut attr), 0);

        return if ret != 0 {
            // The thread failed to start and as a result p was not consumed. Therefore, it is
            // safe to reconstruct the box so that it gets deallocated.
            drop(Box::from_raw(p));
            Err(io::Error::from_raw_os_error(ret))
        } else {
            Ok(Thread { id: native })
        };

        extern "C" fn thread_start(main: *mut libc::c_void) -> *mut libc::c_void {
            unsafe {
                // Next, set up our stack overflow handler which may get triggered if we run
                // out of stack.
                let _handler = stack_overflow::Handler::new();
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

    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn set_name(name: &CStr) {
        const PR_SET_NAME: libc::c_int = 15;
        // pthread wrapper only appeared in glibc 2.12, so we use syscall
        // directly.
        unsafe {
            libc::prctl(PR_SET_NAME, name.as_ptr() as libc::c_ulong, 0, 0, 0);
        }
    }

    #[cfg(any(target_os = "freebsd", target_os = "dragonfly", target_os = "openbsd"))]
    pub fn set_name(name: &CStr) {
        unsafe {
            libc::pthread_set_name_np(libc::pthread_self(), name.as_ptr());
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn set_name(name: &CStr) {
        unsafe {
            libc::pthread_setname_np(name.as_ptr());
        }
    }

    #[cfg(target_os = "netbsd")]
    pub fn set_name(name: &CStr) {
        use crate::ffi::CString;
        let cname = CString::new(&b"%s"[..]).unwrap();
        unsafe {
            libc::pthread_setname_np(
                libc::pthread_self(),
                cname.as_ptr(),
                name.as_ptr() as *mut libc::c_void,
            );
        }
    }

    #[cfg(any(target_os = "solaris", target_os = "illumos"))]
    pub fn set_name(name: &CStr) {
        weak! {
            fn pthread_setname_np(
                libc::pthread_t, *const libc::c_char
            ) -> libc::c_int
        }

        if let Some(f) = pthread_setname_np.get() {
            unsafe {
                f(libc::pthread_self(), name.as_ptr());
            }
        }
    }

    #[cfg(any(
        target_env = "newlib",
        target_os = "haiku",
        target_os = "l4re",
        target_os = "emscripten",
        target_os = "redox",
        target_os = "vxworks"
    ))]
    pub fn set_name(_name: &CStr) {
        // Newlib, Haiku, Emscripten, and VxWorks have no way to set a thread name.
    }
    #[cfg(target_os = "fuchsia")]
    pub fn set_name(_name: &CStr) {
        // FIXME: determine whether Fuchsia has a way to set a thread name.
    }

    pub fn sleep(dur: Duration) {
        let mut secs = dur.as_secs();
        let mut nsecs = dur.subsec_nanos() as _;

        // If we're awoken with a signal then the return value will be -1 and
        // nanosleep will fill in `ts` with the remaining time.
        unsafe {
            while secs > 0 || nsecs > 0 {
                let mut ts = libc::timespec {
                    tv_sec: cmp::min(libc::time_t::MAX as u64, secs) as libc::time_t,
                    tv_nsec: nsecs,
                };
                secs -= ts.tv_sec as u64;
                let ts_ptr = &mut ts as *mut _;
                if libc::nanosleep(ts_ptr, ts_ptr) == -1 {
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
            assert!(ret == 0, "failed to join thread: {}", io::Error::from_raw_os_error(ret));
        }
    }

    pub fn id(&self) -> libc::pthread_t {
        self.id
    }

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

pub fn available_concurrency() -> io::Result<NonZeroUsize> {
    cfg_if::cfg_if! {
        if #[cfg(any(
            target_os = "android",
            target_os = "emscripten",
            target_os = "fuchsia",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "solaris",
            target_os = "illumos",
        ))] {
            match unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) } {
                -1 => Err(io::Error::last_os_error()),
                0 => Err(io::Error::new_const(io::ErrorKind::NotFound, &"The number of hardware threads is not known for the target platform")),
                cpus => Ok(unsafe { NonZeroUsize::new_unchecked(cpus as usize) }),
            }
        } else if #[cfg(any(target_os = "freebsd", target_os = "dragonfly", target_os = "netbsd"))] {
            use crate::ptr;

            let mut cpus: libc::c_uint = 0;
            let mut cpus_size = crate::mem::size_of_val(&cpus);

            unsafe {
                cpus = libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as libc::c_uint;
            }

            // Fallback approach in case of errors or no hardware threads.
            if cpus < 1 {
                let mut mib = [libc::CTL_HW, libc::HW_NCPU, 0, 0];
                let res = unsafe {
                    libc::sysctl(
                        mib.as_mut_ptr(),
                        2,
                        &mut cpus as *mut _ as *mut _,
                        &mut cpus_size as *mut _ as *mut _,
                        ptr::null_mut(),
                        0,
                    )
                };

                // Handle errors if any.
                if res == -1 {
                    return Err(io::Error::last_os_error());
                } else if cpus == 0 {
                    return Err(io::Error::new_const(io::ErrorKind::NotFound, &"The number of hardware threads is not known for the target platform"));
                }
            }
            Ok(unsafe { NonZeroUsize::new_unchecked(cpus as usize) })
        } else if #[cfg(target_os = "openbsd")] {
            use crate::ptr;

            let mut cpus: libc::c_uint = 0;
            let mut cpus_size = crate::mem::size_of_val(&cpus);
            let mut mib = [libc::CTL_HW, libc::HW_NCPU, 0, 0];

            let res = unsafe {
                libc::sysctl(
                    mib.as_mut_ptr(),
                    2,
                    &mut cpus as *mut _ as *mut _,
                    &mut cpus_size as *mut _ as *mut _,
                    ptr::null_mut(),
                    0,
                )
            };

            // Handle errors if any.
            if res == -1 {
                return Err(io::Error::last_os_error());
            } else if cpus == 0 {
                return Err(io::Error::new_const(io::ErrorKind::NotFound, &"The number of hardware threads is not known for the target platform"));
            }

            Ok(unsafe { NonZeroUsize::new_unchecked(cpus as usize) })
        } else {
            // FIXME: implement on vxWorks, Redox, Haiku, l4re
            Err(io::Error::new_const(io::ErrorKind::NotFound, &"The number of hardware threads is not known for the target platform"))
        }
    }
}

#[cfg(all(
    not(target_os = "linux"),
    not(target_os = "freebsd"),
    not(target_os = "macos"),
    not(target_os = "netbsd"),
    not(target_os = "openbsd"),
    not(target_os = "solaris")
))]
#[cfg_attr(test, allow(dead_code))]
pub mod guard {
    use crate::ops::Range;
    pub type Guard = Range<usize>;
    pub unsafe fn current() -> Option<Guard> {
        None
    }
    pub unsafe fn init() -> Option<Guard> {
        None
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "macos",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "solaris"
))]
#[cfg_attr(test, allow(dead_code))]
pub mod guard {
    use libc::{mmap, mprotect};
    use libc::{MAP_ANON, MAP_FAILED, MAP_FIXED, MAP_PRIVATE, PROT_NONE, PROT_READ, PROT_WRITE};

    use crate::io;
    use crate::ops::Range;
    use crate::sync::atomic::{AtomicUsize, Ordering};
    use crate::sys::os;

    // This is initialized in init() and only read from after
    static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

    pub type Guard = Range<usize>;

    #[cfg(target_os = "solaris")]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let mut current_stack: libc::stack_t = crate::mem::zeroed();
        assert_eq!(libc::stack_getbounds(&mut current_stack), 0);
        Some(current_stack.ss_sp)
    }

    #[cfg(target_os = "macos")]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let th = libc::pthread_self();
        let stackaddr =
            libc::pthread_get_stackaddr_np(th) as usize - libc::pthread_get_stacksize_np(th);
        Some(stackaddr as *mut libc::c_void)
    }

    #[cfg(target_os = "openbsd")]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let mut current_stack: libc::stack_t = crate::mem::zeroed();
        assert_eq!(libc::pthread_stackseg_np(libc::pthread_self(), &mut current_stack), 0);

        let stackaddr = if libc::pthread_main_np() == 1 {
            // main thread
            current_stack.ss_sp as usize - current_stack.ss_size + PAGE_SIZE.load(Ordering::Relaxed)
        } else {
            // new thread
            current_stack.ss_sp as usize - current_stack.ss_size
        };
        Some(stackaddr as *mut libc::c_void)
    }

    #[cfg(any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "l4re"
    ))]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let mut ret = None;
        let mut attr: libc::pthread_attr_t = crate::mem::zeroed();
        #[cfg(target_os = "freebsd")]
        assert_eq!(libc::pthread_attr_init(&mut attr), 0);
        #[cfg(target_os = "freebsd")]
        let e = libc::pthread_attr_get_np(libc::pthread_self(), &mut attr);
        #[cfg(not(target_os = "freebsd"))]
        let e = libc::pthread_getattr_np(libc::pthread_self(), &mut attr);
        if e == 0 {
            let mut stackaddr = crate::ptr::null_mut();
            let mut stacksize = 0;
            assert_eq!(libc::pthread_attr_getstack(&attr, &mut stackaddr, &mut stacksize), 0);
            ret = Some(stackaddr);
        }
        if e == 0 || cfg!(target_os = "freebsd") {
            assert_eq!(libc::pthread_attr_destroy(&mut attr), 0);
        }
        ret
    }

    // Precondition: PAGE_SIZE is initialized.
    unsafe fn get_stack_start_aligned() -> Option<*mut libc::c_void> {
        let page_size = PAGE_SIZE.load(Ordering::Relaxed);
        assert!(page_size != 0);
        let stackaddr = get_stack_start()?;

        // Ensure stackaddr is page aligned! A parent process might
        // have reset RLIMIT_STACK to be non-page aligned. The
        // pthread_attr_getstack() reports the usable stack area
        // stackaddr < stackaddr + stacksize, so if stackaddr is not
        // page-aligned, calculate the fix such that stackaddr <
        // new_page_aligned_stackaddr < stackaddr + stacksize
        let remainder = (stackaddr as usize) % page_size;
        Some(if remainder == 0 {
            stackaddr
        } else {
            ((stackaddr as usize) + page_size - remainder) as *mut libc::c_void
        })
    }

    pub unsafe fn init() -> Option<Guard> {
        let page_size = os::page_size();
        PAGE_SIZE.store(page_size, Ordering::Relaxed);

        if cfg!(all(target_os = "linux", not(target_env = "musl"))) {
            // Linux doesn't allocate the whole stack right away, and
            // the kernel has its own stack-guard mechanism to fault
            // when growing too close to an existing mapping.  If we map
            // our own guard, then the kernel starts enforcing a rather
            // large gap above that, rendering much of the possible
            // stack space useless.  See #43052.
            //
            // Instead, we'll just note where we expect rlimit to start
            // faulting, so our handler can report "stack overflow", and
            // trust that the kernel's own stack guard will work.
            let stackaddr = get_stack_start_aligned()?;
            let stackaddr = stackaddr as usize;
            Some(stackaddr - page_size..stackaddr)
        } else if cfg!(all(target_os = "linux", target_env = "musl")) {
            // For the main thread, the musl's pthread_attr_getstack
            // returns the current stack size, rather than maximum size
            // it can eventually grow to. It cannot be used to determine
            // the position of kernel's stack guard.
            None
        } else if cfg!(target_os = "freebsd") {
            // FreeBSD's stack autogrows, and optionally includes a guard page
            // at the bottom.  If we try to remap the bottom of the stack
            // ourselves, FreeBSD's guard page moves upwards.  So we'll just use
            // the builtin guard page.
            let stackaddr = get_stack_start_aligned()?;
            let guardaddr = stackaddr as usize;
            // Technically the number of guard pages is tunable and controlled
            // by the security.bsd.stack_guard_page sysctl, but there are
            // few reasons to change it from the default.  The default value has
            // been 1 ever since FreeBSD 11.1 and 10.4.
            const GUARD_PAGES: usize = 1;
            let guard = guardaddr..guardaddr + GUARD_PAGES * page_size;
            Some(guard)
        } else {
            // Reallocate the last page of the stack.
            // This ensures SIGBUS will be raised on
            // stack overflow.
            // Systems which enforce strict PAX MPROTECT do not allow
            // to mprotect() a mapping with less restrictive permissions
            // than the initial mmap() used, so we mmap() here with
            // read/write permissions and only then mprotect() it to
            // no permissions at all. See issue #50313.
            let stackaddr = get_stack_start_aligned()?;
            let result = mmap(
                stackaddr,
                page_size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON | MAP_FIXED,
                -1,
                0,
            );
            if result != stackaddr || result == MAP_FAILED {
                panic!("failed to allocate a guard page: {}", io::Error::last_os_error());
            }

            let result = mprotect(stackaddr, page_size, PROT_NONE);
            if result != 0 {
                panic!("failed to protect the guard page: {}", io::Error::last_os_error());
            }

            let guardaddr = stackaddr as usize;

            Some(guardaddr..guardaddr + page_size)
        }
    }

    #[cfg(any(target_os = "macos", target_os = "openbsd", target_os = "solaris"))]
    pub unsafe fn current() -> Option<Guard> {
        let stackaddr = get_stack_start()? as usize;
        Some(stackaddr - PAGE_SIZE.load(Ordering::Relaxed)..stackaddr)
    }

    #[cfg(any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "l4re"
    ))]
    pub unsafe fn current() -> Option<Guard> {
        let mut ret = None;
        let mut attr: libc::pthread_attr_t = crate::mem::zeroed();
        #[cfg(target_os = "freebsd")]
        assert_eq!(libc::pthread_attr_init(&mut attr), 0);
        #[cfg(target_os = "freebsd")]
        let e = libc::pthread_attr_get_np(libc::pthread_self(), &mut attr);
        #[cfg(not(target_os = "freebsd"))]
        let e = libc::pthread_getattr_np(libc::pthread_self(), &mut attr);
        if e == 0 {
            let mut guardsize = 0;
            assert_eq!(libc::pthread_attr_getguardsize(&attr, &mut guardsize), 0);
            if guardsize == 0 {
                if cfg!(all(target_os = "linux", target_env = "musl")) {
                    // musl versions before 1.1.19 always reported guard
                    // size obtained from pthread_attr_get_np as zero.
                    // Use page size as a fallback.
                    guardsize = PAGE_SIZE.load(Ordering::Relaxed);
                } else {
                    panic!("there is no guard page");
                }
            }
            let mut stackaddr = crate::ptr::null_mut();
            let mut size = 0;
            assert_eq!(libc::pthread_attr_getstack(&attr, &mut stackaddr, &mut size), 0);

            let stackaddr = stackaddr as usize;
            ret = if cfg!(any(target_os = "freebsd", target_os = "netbsd")) {
                Some(stackaddr - guardsize..stackaddr)
            } else if cfg!(all(target_os = "linux", target_env = "musl")) {
                Some(stackaddr - guardsize..stackaddr)
            } else if cfg!(all(target_os = "linux", target_env = "gnu")) {
                // glibc used to include the guard area within the stack, as noted in the BUGS
                // section of `man pthread_attr_getguardsize`.  This has been corrected starting
                // with glibc 2.27, and in some distro backports, so the guard is now placed at the
                // end (below) the stack.  There's no easy way for us to know which we have at
                // runtime, so we'll just match any fault in the range right above or below the
                // stack base to call that fault a stack overflow.
                Some(stackaddr - guardsize..stackaddr + guardsize)
            } else {
                Some(stackaddr..stackaddr + guardsize)
            };
        }
        if e == 0 || cfg!(target_os = "freebsd") {
            assert_eq!(libc::pthread_attr_destroy(&mut attr), 0);
        }
        ret
    }
}

// glibc >= 2.15 has a __pthread_get_minstack() function that returns
// PTHREAD_STACK_MIN plus bytes needed for thread-local storage.
// We need that information to avoid blowing up when a small stack
// is created in an application with big thread-local storage requirements.
// See #6233 for rationale and details.
#[cfg(target_os = "linux")]
#[allow(deprecated)]
fn min_stack_size(attr: *const libc::pthread_attr_t) -> usize {
    weak!(fn __pthread_get_minstack(*const libc::pthread_attr_t) -> libc::size_t);

    match __pthread_get_minstack.get() {
        None => libc::PTHREAD_STACK_MIN,
        Some(f) => unsafe { f(attr) },
    }
}

// No point in looking up __pthread_get_minstack() on non-glibc
// platforms.
#[cfg(all(not(target_os = "linux"), not(target_os = "netbsd")))]
fn min_stack_size(_: *const libc::pthread_attr_t) -> usize {
    libc::PTHREAD_STACK_MIN
}

#[cfg(target_os = "netbsd")]
fn min_stack_size(_: *const libc::pthread_attr_t) -> usize {
    2048 // just a guess
}
