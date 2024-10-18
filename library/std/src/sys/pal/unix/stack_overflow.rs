#![cfg_attr(test, allow(dead_code))]

pub use self::imp::{cleanup, protect};

#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "hurd",
    target_os = "macos",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "solaris",
    target_os = "illumos",
))]
mod imp {
    use libc::{
        MAP_ANON, MAP_FAILED, MAP_FIXED, MAP_PRIVATE, PROT_NONE, PROT_READ, PROT_WRITE, SA_ONSTACK,
        SA_SIGINFO, SIG_DFL, SIGBUS, SIGSEGV, SS_DISABLE, sigaction, sigaltstack, sighandler_t,
    };
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    use libc::{mmap as mmap64, mprotect, munmap};
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    use libc::{mmap64, mprotect, munmap};

    use crate::ops::Range;
    use crate::sync::OnceLock;
    use crate::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use crate::sys::pal::unix::os;
    use crate::sys::thread_local::{guard, local_pointer};
    use crate::thread::Thread;
    use crate::{io, mem, ptr, thread};

    // We use a TLS variable to store the address of the guard page. While TLS
    // variables are not guaranteed to be signal-safe, this works out in practice
    // since we make sure to write to the variable before the signal stack is
    // installed, thereby ensuring that the variable is always allocated when
    // the signal handler is called.
    local_pointer! {
        static GUARD_START;
        static GUARD_END;

        static SIGALTSTACK;
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
    /// SIGSEGV/SIGBUS entry point
    /// # Safety
    /// Rust doesn't call this, it *gets called*.
    #[forbid(unsafe_op_in_unsafe_fn)]
    unsafe extern "C" fn signal_handler(
        signum: libc::c_int,
        info: *mut libc::siginfo_t,
        _data: *mut libc::c_void,
    ) {
        let start = GUARD_START.get().addr();
        let end = GUARD_END.get().addr();

        // SAFETY: this pointer is provided by the system and will always point to a valid `siginfo_t`.
        let addr = unsafe { (*info).si_addr().addr() };

        // If the faulting address is within the guard page, then we print a
        // message saying so and abort.
        if start <= addr && addr < end {
            let thread = thread::try_current();
            let name = thread.as_ref().and_then(Thread::name).unwrap_or("<unknown>");
            rtprintpanic!("\nthread '{name}' has overflowed its stack\n",);
            rtabort!("stack overflow");
        } else {
            // Unregister ourselves by reverting back to the default behavior.
            // SAFETY: assuming all platforms define struct sigaction as "zero-initializable"
            let mut action: sigaction = unsafe { mem::zeroed() };
            action.sa_sigaction = SIG_DFL;
            // SAFETY: pray this is a well-behaved POSIX implementation of fn sigaction
            unsafe { sigaction(signum, &action, ptr::null_mut()) };

            // See comment above for why this function returns.
        }
    }

    static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);
    static NEED_ALTSTACK: AtomicBool = AtomicBool::new(false);

    /// Set up stack overflow protection for the current thread
    ///
    /// # Safety
    /// May only be called once per thread.
    #[forbid(unsafe_op_in_unsafe_fn)]
    pub unsafe fn protect(main_thread: bool) {
        if main_thread {
            PAGE_SIZE.store(os::page_size(), Ordering::Relaxed);
        // Use acquire ordering to observe the page size store above,
        // which is propagated by a release store to NEED_ALTSTACK.
        } else if !NEED_ALTSTACK.load(Ordering::Acquire) {
            return;
        }

        let guard = if main_thread {
            unsafe { install_main_guard().unwrap_or(0..0) }
        } else {
            unsafe { current_guard().unwrap_or(0..0) }
        };

        // Always store the guard range to ensure the TLS variables are allocated.
        GUARD_START.set(ptr::without_provenance_mut(guard.start));
        GUARD_END.set(ptr::without_provenance_mut(guard.end));

        if main_thread {
            // SAFETY: assuming all platforms define struct sigaction as "zero-initializable"
            let mut action: sigaction = unsafe { mem::zeroed() };
            for &signal in &[SIGSEGV, SIGBUS] {
                // SAFETY: just fetches the current signal handler into action
                unsafe { sigaction(signal, ptr::null_mut(), &mut action) };
                // Configure our signal handler if one is not already set.
                if action.sa_sigaction == SIG_DFL {
                    if !NEED_ALTSTACK.load(Ordering::Relaxed) {
                        // Set up the signal stack and tell other threads to set
                        // up their own. This uses a release store to propagate
                        // the store to PAGE_SIZE above.
                        NEED_ALTSTACK.store(true, Ordering::Release);
                        unsafe { setup_sigaltstack() };
                    }

                    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
                    action.sa_sigaction = signal_handler as sighandler_t;
                    // SAFETY: only overriding signals if the default is set
                    unsafe { sigaction(signal, &action, ptr::null_mut()) };
                }
            }
        } else {
            unsafe { setup_sigaltstack() };
        }
    }

    /// # Safety
    /// Mutates the alternate signal stack
    #[forbid(unsafe_op_in_unsafe_fn)]
    unsafe fn setup_sigaltstack() {
        // SAFETY: assuming stack_t is zero-initializable
        let mut stack = unsafe { mem::zeroed() };
        // SAFETY: reads current stack_t into stack
        unsafe { sigaltstack(ptr::null(), &mut stack) };
        // Do not overwrite the stack if one is already set.
        if stack.ss_flags & SS_DISABLE == 0 {
            return;
        }

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
        let page_size = PAGE_SIZE.load(Ordering::Relaxed);

        let allocation = unsafe {
            mmap64(ptr::null_mut(), sigstack_size + page_size, PROT_READ | PROT_WRITE, flags, -1, 0)
        };
        if allocation == MAP_FAILED {
            panic!("failed to allocate an alternative stack: {}", io::Error::last_os_error());
        }

        let guard_result = unsafe { libc::mprotect(allocation, page_size, PROT_NONE) };
        if guard_result != 0 {
            panic!("failed to set up alternative stack guard page: {}", io::Error::last_os_error());
        }

        let stack = libc::stack_t {
            // Reserve a guard page at the bottom of the allocation.
            ss_sp: unsafe { allocation.add(page_size) },
            ss_flags: 0,
            ss_size: sigstack_size,
        };
        // SAFETY: We warned our caller this would happen!
        unsafe {
            sigaltstack(&stack, ptr::null_mut());
        }

        // Ensure that `rt::thread_cleanup` gets called, which will in turn call
        // cleanup, where this signal stack will be freed.
        guard::enable();
        SIGALTSTACK.set(allocation.cast());
    }

    pub fn cleanup() {
        let allocation = SIGALTSTACK.get();
        if allocation.is_null() {
            return;
        }

        SIGALTSTACK.set(ptr::null_mut());

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
        unsafe { sigaltstack(&disabling_stack, ptr::null_mut()) };

        // SAFETY: we created this mapping in `setup_sigaltstack` above with
        // this exact size.
        unsafe { munmap(allocation.cast(), sigstack_size + page_size) };
    }

    /// Modern kernels on modern hardware can have dynamic signal stack sizes.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn sigstack_size() -> usize {
        let dynamic_sigstksz = unsafe { libc::getauxval(libc::AT_MINSIGSTKSZ) };
        // If getauxval couldn't find the entry, it returns 0,
        // so take the higher of the "constant" and auxval.
        // This transparently supports older kernels which don't provide AT_MINSIGSTKSZ
        libc::SIGSTKSZ.max(dynamic_sigstksz as _)
    }

    /// Not all OS support hardware where this is needed.
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn sigstack_size() -> usize {
        libc::SIGSTKSZ
    }

    #[cfg(any(target_os = "solaris", target_os = "illumos"))]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let mut current_stack: libc::stack_t = crate::mem::zeroed();
        assert_eq!(libc::stack_getbounds(&mut current_stack), 0);
        Some(current_stack.ss_sp)
    }

    #[cfg(target_os = "macos")]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let th = libc::pthread_self();
        let stackptr = libc::pthread_get_stackaddr_np(th);
        Some(stackptr.map_addr(|addr| addr - libc::pthread_get_stacksize_np(th)))
    }

    #[cfg(target_os = "openbsd")]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let mut current_stack: libc::stack_t = crate::mem::zeroed();
        assert_eq!(libc::pthread_stackseg_np(libc::pthread_self(), &mut current_stack), 0);

        let stack_ptr = current_stack.ss_sp;
        let stackaddr = if libc::pthread_main_np() == 1 {
            // main thread
            stack_ptr.addr() - current_stack.ss_size + PAGE_SIZE.load(Ordering::Relaxed)
        } else {
            // new thread
            stack_ptr.addr() - current_stack.ss_size
        };
        Some(stack_ptr.with_addr(stackaddr))
    }

    #[cfg(any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "hurd",
        target_os = "linux",
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

    fn stack_start_aligned(page_size: usize) -> Option<*mut libc::c_void> {
        let stackptr = unsafe { get_stack_start()? };
        let stackaddr = stackptr.addr();

        // Ensure stackaddr is page aligned! A parent process might
        // have reset RLIMIT_STACK to be non-page aligned. The
        // pthread_attr_getstack() reports the usable stack area
        // stackaddr < stackaddr + stacksize, so if stackaddr is not
        // page-aligned, calculate the fix such that stackaddr <
        // new_page_aligned_stackaddr < stackaddr + stacksize
        let remainder = stackaddr % page_size;
        Some(if remainder == 0 {
            stackptr
        } else {
            stackptr.with_addr(stackaddr + page_size - remainder)
        })
    }

    #[forbid(unsafe_op_in_unsafe_fn)]
    unsafe fn install_main_guard() -> Option<Range<usize>> {
        let page_size = PAGE_SIZE.load(Ordering::Relaxed);

        unsafe {
            // this way someone on any unix-y OS can check that all these compile
            if cfg!(all(target_os = "linux", not(target_env = "musl"))) {
                install_main_guard_linux(page_size)
            } else if cfg!(all(target_os = "linux", target_env = "musl")) {
                install_main_guard_linux_musl(page_size)
            } else if cfg!(target_os = "freebsd") {
                install_main_guard_freebsd(page_size)
            } else if cfg!(any(target_os = "netbsd", target_os = "openbsd")) {
                install_main_guard_bsds(page_size)
            } else {
                install_main_guard_default(page_size)
            }
        }
    }

    #[forbid(unsafe_op_in_unsafe_fn)]
    unsafe fn install_main_guard_linux(page_size: usize) -> Option<Range<usize>> {
        // Linux doesn't allocate the whole stack right away, and
        // the kernel has its own stack-guard mechanism to fault
        // when growing too close to an existing mapping. If we map
        // our own guard, then the kernel starts enforcing a rather
        // large gap above that, rendering much of the possible
        // stack space useless. See #43052.
        //
        // Instead, we'll just note where we expect rlimit to start
        // faulting, so our handler can report "stack overflow", and
        // trust that the kernel's own stack guard will work.
        let stackptr = stack_start_aligned(page_size)?;
        let stackaddr = stackptr.addr();
        Some(stackaddr - page_size..stackaddr)
    }

    #[forbid(unsafe_op_in_unsafe_fn)]
    unsafe fn install_main_guard_linux_musl(_page_size: usize) -> Option<Range<usize>> {
        // For the main thread, the musl's pthread_attr_getstack
        // returns the current stack size, rather than maximum size
        // it can eventually grow to. It cannot be used to determine
        // the position of kernel's stack guard.
        None
    }

    #[forbid(unsafe_op_in_unsafe_fn)]
    unsafe fn install_main_guard_freebsd(page_size: usize) -> Option<Range<usize>> {
        // FreeBSD's stack autogrows, and optionally includes a guard page
        // at the bottom. If we try to remap the bottom of the stack
        // ourselves, FreeBSD's guard page moves upwards. So we'll just use
        // the builtin guard page.
        let stackptr = stack_start_aligned(page_size)?;
        let guardaddr = stackptr.addr();
        // Technically the number of guard pages is tunable and controlled
        // by the security.bsd.stack_guard_page sysctl.
        // By default it is 1, checking once is enough since it is
        // a boot time config value.
        static PAGES: OnceLock<usize> = OnceLock::new();

        let pages = PAGES.get_or_init(|| {
            use crate::sys::weak::dlsym;
            dlsym!(fn sysctlbyname(*const libc::c_char, *mut libc::c_void, *mut libc::size_t, *const libc::c_void, libc::size_t) -> libc::c_int);
            let mut guard: usize = 0;
            let mut size = mem::size_of_val(&guard);
            let oid = c"security.bsd.stack_guard_page";
            match sysctlbyname.get() {
                Some(fcn) if unsafe {
                    fcn(oid.as_ptr(),
                        (&raw mut guard).cast(),
                        &raw mut size,
                        ptr::null_mut(),
                        0) == 0
                } => guard,
                _ => 1,
            }
        });
        Some(guardaddr..guardaddr + pages * page_size)
    }

    #[forbid(unsafe_op_in_unsafe_fn)]
    unsafe fn install_main_guard_bsds(page_size: usize) -> Option<Range<usize>> {
        // OpenBSD stack already includes a guard page, and stack is
        // immutable.
        // NetBSD stack includes the guard page.
        //
        // We'll just note where we expect rlimit to start
        // faulting, so our handler can report "stack overflow", and
        // trust that the kernel's own stack guard will work.
        let stackptr = stack_start_aligned(page_size)?;
        let stackaddr = stackptr.addr();
        Some(stackaddr - page_size..stackaddr)
    }

    #[forbid(unsafe_op_in_unsafe_fn)]
    unsafe fn install_main_guard_default(page_size: usize) -> Option<Range<usize>> {
        // Reallocate the last page of the stack.
        // This ensures SIGBUS will be raised on
        // stack overflow.
        // Systems which enforce strict PAX MPROTECT do not allow
        // to mprotect() a mapping with less restrictive permissions
        // than the initial mmap() used, so we mmap() here with
        // read/write permissions and only then mprotect() it to
        // no permissions at all. See issue #50313.
        let stackptr = stack_start_aligned(page_size)?;
        let result = unsafe {
            mmap64(
                stackptr,
                page_size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON | MAP_FIXED,
                -1,
                0,
            )
        };
        if result != stackptr || result == MAP_FAILED {
            panic!("failed to allocate a guard page: {}", io::Error::last_os_error());
        }

        let result = unsafe { mprotect(stackptr, page_size, PROT_NONE) };
        if result != 0 {
            panic!("failed to protect the guard page: {}", io::Error::last_os_error());
        }

        let guardaddr = stackptr.addr();

        Some(guardaddr..guardaddr + page_size)
    }

    #[cfg(any(
        target_os = "macos",
        target_os = "openbsd",
        target_os = "solaris",
        target_os = "illumos",
    ))]
    // FIXME: I am probably not unsafe.
    unsafe fn current_guard() -> Option<Range<usize>> {
        let stackptr = get_stack_start()?;
        let stackaddr = stackptr.addr();
        Some(stackaddr - PAGE_SIZE.load(Ordering::Relaxed)..stackaddr)
    }

    #[cfg(any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "hurd",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "l4re"
    ))]
    // FIXME: I am probably not unsafe.
    unsafe fn current_guard() -> Option<Range<usize>> {
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
            let mut stackptr = crate::ptr::null_mut::<libc::c_void>();
            let mut size = 0;
            assert_eq!(libc::pthread_attr_getstack(&attr, &mut stackptr, &mut size), 0);

            let stackaddr = stackptr.addr();
            ret = if cfg!(any(target_os = "freebsd", target_os = "netbsd", target_os = "hurd")) {
                Some(stackaddr - guardsize..stackaddr)
            } else if cfg!(all(target_os = "linux", target_env = "musl")) {
                Some(stackaddr - guardsize..stackaddr)
            } else if cfg!(all(target_os = "linux", any(target_env = "gnu", target_env = "uclibc")))
            {
                // glibc used to include the guard area within the stack, as noted in the BUGS
                // section of `man pthread_attr_getguardsize`. This has been corrected starting
                // with glibc 2.27, and in some distro backports, so the guard is now placed at the
                // end (below) the stack. There's no easy way for us to know which we have at
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

// This is intentionally not enabled on iOS/tvOS/watchOS/visionOS, as it uses
// several symbols that might lead to rejections from the App Store, namely
// `sigaction`, `sigaltstack`, `sysctlbyname`, `mmap`, `munmap` and `mprotect`.
//
// This might be overly cautious, though it is also what Swift does (and they
// usually have fewer qualms about forwards compatibility, since the runtime
// is shipped with the OS):
// <https://github.com/apple/swift/blob/swift-5.10-RELEASE/stdlib/public/runtime/CrashHandlerMacOS.cpp>
#[cfg(not(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "hurd",
    target_os = "macos",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "solaris",
    target_os = "illumos",
)))]
mod imp {
    pub unsafe fn protect(_main_thread: bool) {}
    pub fn cleanup() {}
}
