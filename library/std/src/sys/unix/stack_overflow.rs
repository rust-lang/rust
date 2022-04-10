#![cfg_attr(test, allow(dead_code))]

use self::imp::{drop_handler, make_handler};

pub use self::imp::cleanup;
pub use self::imp::init;

pub struct Handler {
    data: *mut libc::c_void,
}

impl Handler {
    pub unsafe fn new() -> Handler {
        make_handler()
    }

    fn null() -> Handler {
        Handler { data: crate::ptr::null_mut() }
    }
}

impl Drop for Handler {
    fn drop(&mut self) {
        unsafe {
            drop_handler(self.data);
        }
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "netbsd",
    target_os = "openbsd"
))]
mod imp {
    use super::Handler;
    use crate::io;
    use crate::mem;
    use crate::ptr;
    use crate::thread;

    use libc::MAP_FAILED;
    use libc::{mmap, munmap};
    use libc::{sigaction, sighandler_t, SA_ONSTACK, SA_SIGINFO, SIGBUS, SIG_DFL};
    use libc::{sigaltstack, SIGSTKSZ, SS_DISABLE};
    use libc::{MAP_ANON, MAP_PRIVATE, PROT_NONE, PROT_READ, PROT_WRITE, SIGSEGV};

    use crate::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
    use crate::sys::unix::os::page_size;
    use crate::sys_common::thread_info;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    unsafe fn siginfo_si_addr(info: *mut libc::siginfo_t) -> usize {
        #[repr(C)]
        struct siginfo_t {
            a: [libc::c_int; 3], // si_signo, si_errno, si_code
            si_addr: *mut libc::c_void,
        }

        (*(info as *const siginfo_t)).si_addr as usize
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    unsafe fn siginfo_si_addr(info: *mut libc::siginfo_t) -> usize {
        (*info).si_addr as usize
    }

    // Signal handler for the SIGSEGV and SIGBUS handlers. We've got guard pages
    // (unmapped pages) at the end of every thread's stack, so if a thread ends
    // up running into the guard page it'll trigger this handler. We want to
    // detect these cases and print out a helpful error saying that the stack
    // has overflowed. All other signals, however, should go back to what they
    // were originally supposed to do.
    //
    // This handler currently exists purely to print an informative message
    // whenever a thread overflows its stack. We then abort to exit and
    // indicate a crash, but to avoid a misleading SIGSEGV that might lead
    // users to believe that unsafe code has accessed an invalid pointer; the
    // SIGSEGV encountered when overflowing the stack is expected and
    // well-defined.
    //
    // If this is not a stack overflow, the handler un-registers itself and
    // then returns (to allow the original signal to be delivered again).
    // Returning from this kind of signal handler is technically not defined
    // to work when reading the POSIX spec strictly, but in practice it turns
    // out many large systems and all implementations allow returning from a
    // signal handler to work. For a more detailed explanation see the
    // comments on #26458.
    unsafe extern "C" fn signal_handler(
        signum: libc::c_int,
        info: *mut libc::siginfo_t,
        _data: *mut libc::c_void,
    ) {
        let guard = thread_info::stack_guard().unwrap_or(0..0);
        let addr = siginfo_si_addr(info);

        // If the faulting address is within the guard page, then we print a
        // message saying so and abort.
        if guard.start <= addr && addr < guard.end {
            rtprintpanic!(
                "\nthread '{}' has overflowed its stack\n",
                thread::current().name().unwrap_or("<unknown>")
            );
            rtabort!("stack overflow");
        } else {
            // Unregister ourselves by reverting back to the default behavior.
            let mut action: sigaction = mem::zeroed();
            action.sa_sigaction = SIG_DFL;
            sigaction(signum, &action, ptr::null_mut());

            // See comment above for why this function returns.
        }
    }

    static MAIN_ALTSTACK: AtomicPtr<libc::c_void> = AtomicPtr::new(ptr::null_mut());
    static NEED_ALTSTACK: AtomicBool = AtomicBool::new(false);

    pub unsafe fn init() {
        let mut action: sigaction = mem::zeroed();
        for &signal in &[SIGSEGV, SIGBUS] {
            sigaction(signal, ptr::null_mut(), &mut action);
            // Configure our signal handler if one is not already set.
            if action.sa_sigaction == SIG_DFL {
                action.sa_flags = SA_SIGINFO | SA_ONSTACK;
                action.sa_sigaction = signal_handler as sighandler_t;
                sigaction(signal, &action, ptr::null_mut());
                NEED_ALTSTACK.store(true, Ordering::Relaxed);
            }
        }

        let handler = make_handler();
        MAIN_ALTSTACK.store(handler.data, Ordering::Relaxed);
        mem::forget(handler);
    }

    pub unsafe fn cleanup() {
        drop_handler(MAIN_ALTSTACK.load(Ordering::Relaxed));
    }

    unsafe fn get_stackp() -> *mut libc::c_void {
        // OpenBSD requires this flag for stack mapping
        // otherwise the said mapping will fail as a no-op on most systems
        // and has a different meaning on FreeBSD
        #[cfg(any(target_os = "openbsd", target_os = "netbsd", target_os = "linux",))]
        let flags = MAP_PRIVATE | MAP_ANON | libc::MAP_STACK;
        #[cfg(not(any(target_os = "openbsd", target_os = "netbsd", target_os = "linux",)))]
        let flags = MAP_PRIVATE | MAP_ANON;
        let stackp =
            mmap(ptr::null_mut(), SIGSTKSZ + page_size(), PROT_READ | PROT_WRITE, flags, -1, 0);
        if stackp == MAP_FAILED {
            panic!("failed to allocate an alternative stack: {}", io::Error::last_os_error());
        }
        let guard_result = libc::mprotect(stackp, page_size(), PROT_NONE);
        if guard_result != 0 {
            panic!("failed to set up alternative stack guard page: {}", io::Error::last_os_error());
        }
        stackp.add(page_size())
    }

    unsafe fn get_stack() -> libc::stack_t {
        libc::stack_t { ss_sp: get_stackp(), ss_flags: 0, ss_size: SIGSTKSZ }
    }

    pub unsafe fn make_handler() -> Handler {
        if !NEED_ALTSTACK.load(Ordering::Relaxed) {
            return Handler::null();
        }
        let mut stack = mem::zeroed();
        sigaltstack(ptr::null(), &mut stack);
        // Configure alternate signal stack, if one is not already set.
        if stack.ss_flags & SS_DISABLE != 0 {
            stack = get_stack();
            sigaltstack(&stack, ptr::null_mut());
            Handler { data: stack.ss_sp as *mut libc::c_void }
        } else {
            Handler::null()
        }
    }

    pub unsafe fn drop_handler(data: *mut libc::c_void) {
        if !data.is_null() {
            let stack = libc::stack_t {
                ss_sp: ptr::null_mut(),
                ss_flags: SS_DISABLE,
                // Workaround for bug in macOS implementation of sigaltstack
                // UNIX2003 which returns ENOMEM when disabling a stack while
                // passing ss_size smaller than MINSIGSTKSZ. According to POSIX
                // both ss_sp and ss_size should be ignored in this case.
                ss_size: SIGSTKSZ,
            };
            sigaltstack(&stack, ptr::null_mut());
            // We know from `get_stackp` that the alternate stack we installed is part of a mapping
            // that started one page earlier, so walk back a page and unmap from there.
            munmap(data.sub(page_size()), SIGSTKSZ + page_size());
        }
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "netbsd",
    target_os = "openbsd",
)))]
mod imp {
    pub unsafe fn init() {}

    pub unsafe fn cleanup() {}

    pub unsafe fn make_handler() -> super::Handler {
        super::Handler::null()
    }

    pub unsafe fn drop_handler(_data: *mut libc::c_void) {}
}
