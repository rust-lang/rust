#![cfg_attr(test, allow(dead_code))]

use self::imp::{make_handler, drop_handler};

pub use self::imp::cleanup;
pub use self::imp::init;

pub struct Handler {
    _data: *mut libc::c_void
}

impl Handler {
    pub unsafe fn new() -> Handler {
        make_handler()
    }
}

impl Drop for Handler {
    fn drop(&mut self) {
        unsafe {
            drop_handler(self);
        }
    }
}

#[cfg(any(target_os = "linux",
          target_os = "macos",
          target_os = "dragonfly",
          target_os = "freebsd",
          target_os = "solaris",
          all(target_os = "netbsd", not(target_vendor = "rumprun")),
          target_os = "openbsd"))]
mod imp {
    use super::Handler;
    use crate::mem;
    use crate::ptr;

    use libc::{sigaltstack, SIGSTKSZ, SS_DISABLE};
    use libc::{sigaction, SIGBUS, SIG_DFL,
               SA_SIGINFO, SA_ONSTACK, sighandler_t};
    use libc::{mmap, munmap};
    use libc::{SIGSEGV, PROT_READ, PROT_WRITE, MAP_PRIVATE, MAP_ANON};
    use libc::MAP_FAILED;

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
    unsafe extern fn signal_handler(signum: libc::c_int,
                                    info: *mut libc::siginfo_t,
                                    _data: *mut libc::c_void) {
        use crate::sys_common::util::report_overflow;

        let guard = thread_info::stack_guard().unwrap_or(0..0);
        let addr = siginfo_si_addr(info);

        // If the faulting address is within the guard page, then we print a
        // message saying so and abort.
        if guard.start <= addr && addr < guard.end {
            report_overflow();
            rtabort!("stack overflow");
        } else {
            // Unregister ourselves by reverting back to the default behavior.
            let mut action: sigaction = mem::zeroed();
            action.sa_sigaction = SIG_DFL;
            sigaction(signum, &action, ptr::null_mut());

            // See comment above for why this function returns.
        }
    }

    static mut MAIN_ALTSTACK: *mut libc::c_void = ptr::null_mut();

    pub unsafe fn init() {
        let mut action: sigaction = mem::zeroed();
        action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        action.sa_sigaction = signal_handler as sighandler_t;
        sigaction(SIGSEGV, &action, ptr::null_mut());
        sigaction(SIGBUS, &action, ptr::null_mut());

        let handler = make_handler();
        MAIN_ALTSTACK = handler._data;
        mem::forget(handler);
    }

    pub unsafe fn cleanup() {
        Handler { _data: MAIN_ALTSTACK };
    }

    unsafe fn get_stackp() -> *mut libc::c_void {
        let stackp = mmap(ptr::null_mut(),
                          SIGSTKSZ,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON,
                          -1,
                          0);
        if stackp == MAP_FAILED {
            panic!("failed to allocate an alternative stack");
        }
        stackp
    }

    #[cfg(any(target_os = "linux",
              target_os = "macos",
              target_os = "freebsd",
              target_os = "netbsd",
              target_os = "openbsd",
              target_os = "solaris"))]
    unsafe fn get_stack() -> libc::stack_t {
        libc::stack_t { ss_sp: get_stackp(), ss_flags: 0, ss_size: SIGSTKSZ }
    }

    #[cfg(target_os = "dragonfly")]
    unsafe fn get_stack() -> libc::stack_t {
        libc::stack_t { ss_sp: get_stackp() as *mut i8, ss_flags: 0, ss_size: SIGSTKSZ }
    }

    pub unsafe fn make_handler() -> Handler {
        let mut stack = mem::zeroed();
        sigaltstack(ptr::null(), &mut stack);
        // Configure alternate signal stack, if one is not already set.
        if stack.ss_flags & SS_DISABLE != 0 {
            stack = get_stack();
            sigaltstack(&stack, ptr::null_mut());
            Handler { _data: stack.ss_sp as *mut libc::c_void }
        } else {
            Handler { _data: ptr::null_mut() }
        }
    }

    pub unsafe fn drop_handler(handler: &mut Handler) {
        if !handler._data.is_null() {
            let stack =  libc::stack_t {
                ss_sp: ptr::null_mut(),
                ss_flags: SS_DISABLE,
                // Workaround for bug in macOS implementation of sigaltstack
                // UNIX2003 which returns ENOMEM when disabling a stack while
                // passing ss_size smaller than MINSIGSTKSZ. According to POSIX
                // both ss_sp and ss_size should be ignored in this case.
                ss_size: SIGSTKSZ,
            };
            sigaltstack(&stack, ptr::null_mut());
            munmap(handler._data, SIGSTKSZ);
        }
    }
}

#[cfg(not(any(target_os = "linux",
              target_os = "macos",
              target_os = "dragonfly",
              target_os = "freebsd",
              target_os = "solaris",
              all(target_os = "netbsd", not(target_vendor = "rumprun")),
              target_os = "openbsd")))]
mod imp {
    use crate::ptr;

    pub unsafe fn init() {
    }

    pub unsafe fn cleanup() {
    }

    pub unsafe fn make_handler() -> super::Handler {
        super::Handler { _data: ptr::null_mut() }
    }

    pub unsafe fn drop_handler(_handler: &mut super::Handler) {
    }
}
