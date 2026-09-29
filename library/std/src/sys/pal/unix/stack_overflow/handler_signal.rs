use libc::{
    MAP_ANON, MAP_FAILED, MAP_PRIVATE, PROT_NONE, PROT_READ, PROT_WRITE, SA_ONSTACK, SA_SIGINFO,
    SIG_DFL, SIGBUS, SIGSEGV, SS_DISABLE, sigaction, sigaltstack, sighandler_t,
};
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
use libc::{mmap as mmap64, mprotect, munmap};
#[cfg(all(target_os = "linux", target_env = "gnu"))]
use libc::{mmap64, mprotect, munmap};

use super::Handler;
use super::thread_info::{delete_current_info, set_current_info, with_current_info};
use crate::ops::Range;
use crate::sync::atomic::{Atomic, AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use crate::sys::pal::unix::conf;
use crate::{io, mem, ptr};

/// Signal handler for the SIGSEGV and SIGBUS handlers.
///
/// We've got guard pages (unmapped pages) at the end of every thread's
/// stack, so if a thread ends up running into the guard page it'll trigger
/// this handler. We want to detect these cases and print out a helpful error
/// saying that the stack has overflowed. All other signals, however, should
/// go back to what they were originally supposed to do.
///
/// This handler currently exists purely to print an informative message
/// whenever a thread overflows its stack. We then abort to exit and
/// indicate a crash, but to avoid a misleading SIGSEGV that might lead
/// users to believe that unsafe code has accessed an invalid pointer; the
/// SIGSEGV encountered when overflowing the stack is expected and
/// well-defined.
///
/// If this is not a stack overflow, the handler un-registers itself and
/// then returns (to allow the original signal to be delivered again).
/// Returning from this kind of signal handler is technically not defined
/// to work when reading the POSIX spec strictly, but in practice it turns
/// out many large systems and all implementations allow returning from a
/// signal handler to work. For a more detailed explanation see the
/// comments on #26458.
///
/// # Safety
/// Rust doesn't call this, it *gets called* by the kernel, which we expect
/// to provide valid parameters. Apart from that, this function does not
/// have any other preconditions.
unsafe extern "C" fn signal_handler(
    signum: libc::c_int,
    info: *mut libc::siginfo_t,
    _data: *mut libc::c_void,
) {
    // SAFETY: this pointer is provided by the system and will always point to a valid `siginfo_t`.
    let fault_addr = unsafe { (*info).si_addr().addr() };

    // `with_current_info` expects that the process aborts after it is
    // called. If the signal was not caused by a memory access, this might
    // not be true. We detect this by noticing that the `si_addr` field is
    // zero if the signal is synthetic.
    if fault_addr != 0 {
        with_current_info(|thread_info| {
            // If the faulting address is within the guard page, then we print a
            // message saying so and abort.
            if let Some(thread_info) = thread_info
                && thread_info.guard_page_range.contains(&fault_addr)
            {
                // Hey you! Yes, you modifying the stack overflow message!
                // Please make sure that all functions called here are
                // actually async-signal-safe. If they're not, try retrieving
                // the information beforehand and storing it in `ThreadInfo`.
                // Thank you!
                // - says Jonas after having had to watch his carefully
                //   written code get made unsound again.
                let tid = thread_info.tid;
                let name = thread_info.name.as_deref().unwrap_or("<unknown>");
                rtprintpanic!("\nthread '{name}' ({tid}) has overflowed its stack\n");
                rtabort!("stack overflow");
            }
        })
    }

    // Unregister ourselves by reverting back to the default behavior.
    // SAFETY: assuming all platforms define struct sigaction as "zero-initializable"
    let mut action: sigaction = unsafe { mem::zeroed() };
    action.sa_sigaction = SIG_DFL;
    // SAFETY: pray this is a well-behaved POSIX implementation of fn sigaction
    unsafe { sigaction(signum, &action, ptr::null_mut()) };

    // See comment above for why this function returns.
}

static PAGE_SIZE: Atomic<usize> = AtomicUsize::new(0);
// Store a pointer to the allocation for the main thread's altstack so that
// tools like valgrind don't complain about a leaked unreachable allocation.
//
// If the main thread exits, the process will terminate so there's no use in
// freeing resources. It also means that the altstack is still installed
// while TLS destructors are run on the main thread (c.f. #111272).
static MAIN_ALTSTACK: Atomic<*mut libc::c_void> = AtomicPtr::new(ptr::null_mut());
static NEED_ALTSTACK: Atomic<bool> = AtomicBool::new(false);

pub fn init(guard_page_range: Option<Range<usize>>) {
    let page_size = conf::page_size();
    PAGE_SIZE.store(page_size, Ordering::Relaxed);

    let mut guard_page_range =
        guard_page_range.or_else(|| super::guard_page::find_main_guard(page_size));

    // SAFETY: C structures are always zero-initializable.
    let mut action: sigaction = unsafe { mem::zeroed() };
    for &signal in &[SIGSEGV, SIGBUS] {
        // SAFETY: just fetches the current signal handler into action
        unsafe { sigaction(signal, ptr::null_mut(), &mut action) };
        // We assume that overriding the signal handler is always safe,
        // which might conflict with certain libraries that rely on a
        // specific signal behaviour. To prevent problems, we only
        // override the handler if it has not been set yet.
        if action.sa_sigaction == SIG_DFL {
            if !NEED_ALTSTACK.load(Ordering::Relaxed) {
                // haven't set up our sigaltstack yet
                NEED_ALTSTACK.store(true, Ordering::Release);
                let handler = make_handler(true);
                MAIN_ALTSTACK.store(handler.data, Ordering::Relaxed);
                mem::forget(handler);

                if let Some(guard_page_range) = guard_page_range.take() {
                    set_current_info(guard_page_range);
                }
            }

            action.sa_flags = SA_SIGINFO | SA_ONSTACK;
            action.sa_sigaction = signal_handler
                as unsafe extern "C" fn(i32, *mut libc::siginfo_t, *mut libc::c_void)
                as sighandler_t;
            // SAFETY:
            // `&action` describes a valid `sigaction` and `signal_handler`
            // is safe to use as a signal handler for `SIGSEGV` and `SIGBUS`.
            unsafe { sigaction(signal, &action, ptr::null_mut()) };
        }
    }
}

fn get_stack(page_size: usize) -> libc::stack_t {
    // OpenBSD requires this flag for stack mapping
    // otherwise the said mapping will fail as a no-op on most systems
    // and has a different meaning on FreeBSD
    #[cfg(any(
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "linux",
        target_os = "dragonfly",
    ))]
    let flags = MAP_PRIVATE | MAP_ANON | libc::MAP_STACK;
    #[cfg(not(any(
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "linux",
        target_os = "dragonfly",
    )))]
    let flags = MAP_PRIVATE | MAP_ANON;

    let sigstack_size = sigstack_size();

    // SAFETY: this does not unmap any existing pages.
    let stackp = unsafe {
        mmap64(ptr::null_mut(), sigstack_size + page_size, PROT_READ | PROT_WRITE, flags, -1, 0)
    };
    if stackp == MAP_FAILED {
        panic!("failed to allocate an alternative stack: {}", io::Error::last_os_error());
    }
    // SAFETY: this only affects the memory we just allocated.
    let guard_result = unsafe { mprotect(stackp, page_size, PROT_NONE) };
    if guard_result != 0 {
        panic!("failed to set up alternative stack guard page: {}", io::Error::last_os_error());
    }
    // SAFETY:
    // The region was allocated with a larger size than `page_size`, so this
    // addition is within bounds.
    let stackp = unsafe { stackp.add(page_size) };

    libc::stack_t { ss_sp: stackp, ss_flags: 0, ss_size: sigstack_size }
}

pub fn make_handler(main_thread: bool) -> Handler {
    if cfg!(panic = "immediate-abort") || !NEED_ALTSTACK.load(Ordering::Acquire) {
        return Handler::null();
    }

    let page_size = PAGE_SIZE.load(Ordering::Relaxed);

    if !main_thread {
        if let Some(guard_page_range) = super::guard_page::current_guard(page_size) {
            set_current_info(guard_page_range);
        }
    }

    // Load the current alternate signal stack to see if we need to install
    // our own.
    //
    // SAFETY: C structures are always zero-initializable.
    let mut stack = unsafe { mem::zeroed() };
    // SAFETY: `&mut stack` is valid for writing a `stack_t`.
    unsafe { sigaltstack(ptr::null(), &mut stack) };

    // Configure alternate signal stack, if one is not already set.
    if stack.ss_flags & SS_DISABLE != 0 {
        let stack = get_stack(page_size);
        // SAFETY:
        // `stack_t` is a freshly allocated stack that's not used anywhere
        // else. It contains a guard page, so stack overflows in signal
        // handlers will not cause undefined behaviour. We must make the
        // fundamental runtime assumption that it is safe to install an
        // alternate signal stack if there is none currently installed.
        // This might conflict with foreign libraries that use the existence
        // of an alternate signal stack as indication that certain runtime
        // initialisation by the library has been performed (e.g. old
        // versions of `std` assumed that certain thread-locals were already
        // accessed and thus initialized in the thread if the stack overflow
        // signal was successfully delivered). Such assumptions in other
        // libraries are fundamentally flawed, so we pay no regard to them.
        unsafe { sigaltstack(&stack, ptr::null_mut()) };
        Handler { data: stack.ss_sp as *mut libc::c_void }
    } else {
        Handler::null()
    }
}

/// # Safety
/// Must only be called with a pointer returned by `make_handler`, and only
/// once per `Handler`.
pub unsafe fn drop_handler(data: *mut libc::c_void) {
    if !data.is_null() {
        let sigstack_size = sigstack_size();
        let page_size = PAGE_SIZE.load(Ordering::Relaxed);
        let disabling_stack = libc::stack_t {
            ss_sp: ptr::null_mut(),
            ss_flags: SS_DISABLE,
            // Workaround for bug in macOS implementation of sigaltstack
            // UNIX2003 which returns ENOMEM when disabling a stack while
            // passing ss_size smaller than MINSIGSTKSZ. According to POSIX
            // both ss_sp and ss_size should be ignored in this case.
            ss_size: sigstack_size,
        };
        // SAFETY:
        // We assume that disabling the alternate signal stack is always
        // sound, even if the current alternate signal stack is not the one
        // we installed in `make_handler`. Any stack overflows from this
        // point on will abort the program when the kernel tries to write
        // the signal information to the guard page.
        //
        // FIXME: detect if the stack has changed, and only uninstall if it hasn't.
        unsafe { sigaltstack(&disabling_stack, ptr::null_mut()) };
        // The stack returned by `get_stack` is part of a mapping that
        // started one page earlier, so walk back a page and unmap from
        // there.
        //
        // SAFETY:
        // This allocation was created by us in `get_stack` and, as the
        // alternate signal stack is now disabled, is no longer in use.
        unsafe { munmap(data.sub(page_size), sigstack_size + page_size) };
    }

    delete_current_info();
}

/// Modern kernels on modern hardware can have dynamic signal stack sizes.
#[cfg(all(any(target_os = "linux", target_os = "android"), not(target_env = "uclibc")))]
fn sigstack_size() -> usize {
    // SAFETY: `getauxval` is always safe to call.
    let dynamic_sigstksz = unsafe { libc::getauxval(libc::AT_MINSIGSTKSZ) };
    // If getauxval couldn't find the entry, it returns 0,
    // so take the higher of the "constant" and auxval.
    // This transparently supports older kernels which don't provide AT_MINSIGSTKSZ
    libc::SIGSTKSZ.max(dynamic_sigstksz as _)
}

/// Not all OS support hardware where this is needed.
#[cfg(not(all(any(target_os = "linux", target_os = "android"), not(target_env = "uclibc"))))]
fn sigstack_size() -> usize {
    libc::SIGSTKSZ
}
