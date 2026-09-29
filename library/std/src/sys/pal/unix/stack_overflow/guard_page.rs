use crate::ops::Range;

#[cfg(any(target_os = "solaris", target_os = "illumos"))]
fn get_stack_start(_page_size: usize) -> Option<*mut libc::c_void> {
    // SAFETY: C types are always zero-initializable.
    let mut current_stack: libc::stack_t = unsafe { crate::mem::zeroed() };
    // SAFETY:
    // The pointer is valid for writing a `stack_t`.
    assert_eq!(unsafe { libc::stack_getbounds(&mut current_stack) }, 0);
    Some(current_stack.ss_sp)
}

#[cfg(target_os = "macos")]
fn get_stack_start(_page_size: usize) -> Option<*mut libc::c_void> {
    // SAFETY: always safe to call.
    let th = unsafe { libc::pthread_self() };
    // SAFETY: `th` is a valid `pthread_t`.
    unsafe {
        let stackptr = libc::pthread_get_stackaddr_np(th);
        let stacksize = libc::pthread_get_stacksize_np(th);
        Some(stackptr.map_addr(|addr| addr - stacksize))
    }
}

#[cfg(target_os = "openbsd")]
fn get_stack_start(page_size: usize) -> Option<*mut libc::c_void> {
    // SAFETY: C types are always zero-initializable.
    let mut current_stack: libc::stack_t = unsafe { crate::mem::zeroed() };
    // SAFETY:
    // * calling `pthread_self` is always valid and returns a valid `pthread_t`.
    // * `&mut current_stack` is coerced to a pointer that is valid for writing
    //   a `stack_t`.
    assert_eq!(unsafe { libc::pthread_stackseg_np(libc::pthread_self(), &mut current_stack) }, 0);

    let stack_ptr = current_stack.ss_sp;
    // SAFETY: this is always safe to call.
    let stackaddr = if unsafe { libc::pthread_main_np() } == 1 {
        // main thread
        stack_ptr.addr() - current_stack.ss_size + page_size
    } else {
        // new thread
        stack_ptr.addr() - current_stack.ss_size
    };
    Some(stack_ptr.with_addr(stackaddr))
}

#[cfg(any(
    //target_os = "android", (currently unused)
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "hurd",
    all(target_os = "linux", not(target_env = "musl")),
    //target_os = "l4re" (currently unused)
))]
fn get_stack_start(_page_size: usize) -> Option<*mut libc::c_void> {
    use crate::pin::pin;
    use crate::sys::helpers::COpaque;

    let mut ret = None;
    let mut attr: COpaque<libc::pthread_attr_t> = COpaque::uninit();
    if !cfg!(target_os = "freebsd") {
        attr = COpaque::zeroed();
    }
    let attr = pin!(attr);
    // FIXME(pin-ergonomics): remove the next line.
    let attr = attr.into_ref();

    // SAFETY:
    // The attributes object has not been initialized yet and will not be
    // moved until destroyed.
    #[cfg(target_os = "freebsd")]
    assert_eq!(unsafe { libc::pthread_attr_init(attr.get()) }, 0);
    // SAFETY:
    // * calling `pthread_self` is always valid and returns a valid `pthread_t`
    // * `attr` is an initialized attribute object that can be written to.
    #[cfg(target_os = "freebsd")]
    let e = unsafe { libc::pthread_attr_get_np(libc::pthread_self(), attr.get()) };
    // SAFETY:
    // * calling `pthread_self` is always valid and returns a valid `pthread_t`
    // * `attr` can be written to, and will be initialized by this call.
    #[cfg(not(target_os = "freebsd"))]
    let e = unsafe { libc::pthread_getattr_np(libc::pthread_self(), attr.get()) };
    if e == 0 {
        let mut stackaddr = crate::ptr::null_mut();
        let mut stacksize = 0;
        // SAFETY:
        // `attr` is an initialized attribute object and both the pointers
        // are valid for writing.
        assert_eq!(
            unsafe { libc::pthread_attr_getstack(attr.get(), &mut stackaddr, &mut stacksize) },
            0
        );
        ret = Some(stackaddr);
    }
    if e == 0 || cfg!(target_os = "freebsd") {
        // SAFETY:
        // `attr` was initialized either by `pthread_attr_init` (FreeBSD) or
        // by `pthread_attr_get_np`, and is not used after this point.
        assert_eq!(unsafe { libc::pthread_attr_destroy(attr.get()) }, 0);
    }
    ret
}

#[cfg(any(
    target_os = "hurd",
    target_os = "macos",
    target_os = "solaris",
    target_os = "illumos",
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
))]
fn stack_start_aligned(page_size: usize) -> Option<*mut libc::c_void> {
    let stackptr = get_stack_start(page_size)?;
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

/// # Safety
/// This function must only be called from the main thread, and there must
/// be sufficient stack space remaining to place a stack guard.
pub unsafe fn install_main_guard() -> Option<Range<usize>> {
    cfg_select! {
        any(target_os = "hurd", target_os = "macos", target_os = "solaris", target_os = "illumos",) => {
            use crate::io::Error;

            let page_size = crate::sys::pal::conf::page_size();

            // Reallocate the last page of the stack.
            // This ensures SIGBUS will be raised on
            // stack overflow.
            // Systems which enforce strict PAX MPROTECT do not allow
            // to mprotect() a mapping with less restrictive permissions
            // than the initial mmap() used, so we mmap() here with
            // read/write permissions and only then mprotect() it to
            // no permissions at all. See issue #50313.
            let stackptr = stack_start_aligned(page_size)?;
            // SAFETY:
            // The memory region from `stackptr..stackptr + page_size` belongs to
            // the current thread's stack, and the caller has asserted that there
            // is sufficient stack space, which means that this will not overwrite
            // any existing allocations.
            let result = unsafe {
                libc::mmap(
                    stackptr,
                    page_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
                    -1,
                    0,
                )
            };
            if result != stackptr || result == libc::MAP_FAILED {
                panic!("failed to allocate a guard page: {}", Error::last_os_error());
            }

            // SAFETY:
            // Since this function is only called on the main thread, the stack will
            // not be reused until program exit, so the runtime will never observe
            // that part of the stack has been made unusable in this way.
            let result = unsafe { libc::mprotect(stackptr, page_size, libc::PROT_NONE) };
            if result != 0 {
                panic!("failed to protect the guard page: {}", Error::last_os_error());
            }

            let guardaddr = stackptr.addr();

            Some(guardaddr..guardaddr + page_size)
        }
        _ => None,
    }
}

#[cfg(all(target_os = "linux", not(target_env = "musl")))]
pub fn find_main_guard(page_size: usize) -> Option<Range<usize>> {
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

#[cfg(all(target_os = "linux", target_env = "musl"))]
pub fn find_main_guard(_page_size: usize) -> Option<Range<usize>> {
    // For the main thread, the musl's pthread_attr_getstack
    // returns the current stack size, rather than maximum size
    // it can eventually grow to. It cannot be used to determine
    // the position of kernel's stack guard.
    None
}

#[cfg(target_os = "freebsd")]
pub fn find_main_guard(page_size: usize) -> Option<Range<usize>> {
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
    // FIXME(joboet): this function is only called once, remove the caching.
    static PAGES: crate::sync::OnceLock<usize> = crate::sync::OnceLock::new();

    let pages = PAGES.get_or_init(|| {
        let mut guard: usize = 0;
        let mut size = size_of_val(&guard);
        let oid = c"security.bsd.stack_guard_page";

        let r = unsafe {
            libc::sysctlbyname(
                oid.as_ptr(),
                (&raw mut guard).cast(),
                &raw mut size,
                crate::ptr::null_mut(),
                0,
            )
        };
        if r == 0 { guard } else { 1 }
    });
    Some(guardaddr..guardaddr + pages * page_size)
}

#[cfg(any(target_os = "netbsd", target_os = "openbsd"))]
pub fn find_main_guard(page_size: usize) -> Option<Range<usize>> {
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

#[cfg(any(target_os = "hurd", target_os = "macos", target_os = "solaris", target_os = "illumos",))]
pub fn find_main_guard(_page_size: usize) -> Option<Range<usize>> {
    // We installed the main thread's guard page ourselves in `install_main_guard`,
    // so this function will only be called if that fails, in which case there
    // won't be any guard page.
    None
}

#[cfg(any(
    target_os = "macos",
    target_os = "openbsd",
    target_os = "solaris",
    target_os = "illumos",
))]
pub fn current_guard(page_size: usize) -> Option<Range<usize>> {
    let stackptr = get_stack_start(page_size)?;
    let stackaddr = stackptr.addr();
    Some(stackaddr - page_size..stackaddr)
}

#[cfg(any(
    //target_os = "android", (currently unused)
    target_os = "freebsd",
    target_os = "hurd",
    target_os = "linux",
    target_os = "netbsd",
    //target_os = "l4re" (currently unused)
))]
pub fn current_guard(page_size: usize) -> Option<Range<usize>> {
    use crate::pin::pin;
    use crate::sys::helpers::COpaque;

    let mut ret = None;

    let mut attr: COpaque<libc::pthread_attr_t> = COpaque::uninit();
    if !cfg!(target_os = "freebsd") {
        attr = COpaque::zeroed();
    }
    let attr = pin!(attr);
    // FIXME(pin-ergonomics): remove the next line.
    let attr = attr.into_ref();

    // SAFETY:
    // The attributes object has not been initialized yet and will not be
    // moved until destroyed.
    #[cfg(target_os = "freebsd")]
    assert_eq!(unsafe { libc::pthread_attr_init(attr.get()) }, 0);
    // SAFETY:
    // * calling `pthread_self` is always valid and returns a valid `pthread_t`
    // * `attr` is an initialized attribute object that can be written to.
    #[cfg(target_os = "freebsd")]
    let e = unsafe { libc::pthread_attr_get_np(libc::pthread_self(), attr.get()) };
    // SAFETY:
    // * calling `pthread_self` is always valid and returns a valid `pthread_t`
    // * `attr` can be written to, and will be initialized by this call.
    #[cfg(not(target_os = "freebsd"))]
    let e = unsafe { libc::pthread_getattr_np(libc::pthread_self(), attr.get()) };
    if e == 0 {
        let mut guardsize = 0;
        // SAFETY:
        // `attr` is an initialized attribute object and the pointer is valid
        // for writing.
        assert_eq!(unsafe { libc::pthread_attr_getguardsize(attr.get(), &mut guardsize) }, 0);
        if guardsize == 0 {
            if cfg!(all(target_os = "linux", target_env = "musl")) {
                // musl versions before 1.1.19 always reported guard
                // size obtained from pthread_attr_get_np as zero.
                // Use page size as a fallback.
                guardsize = page_size;
            } else {
                panic!("there is no guard page");
            }
        }
        let mut stackptr = crate::ptr::null_mut::<libc::c_void>();
        let mut size = 0;
        // SAFETY:
        // `attr` is an initialized attribute object and both the pointers
        // are valid for writing.
        assert_eq!(unsafe { libc::pthread_attr_getstack(attr.get(), &mut stackptr, &mut size) }, 0);

        let stackaddr = stackptr.addr();
        ret = if cfg!(any(target_os = "freebsd", target_os = "netbsd", target_os = "hurd")) {
            Some(stackaddr - guardsize..stackaddr)
        } else if cfg!(all(target_os = "linux", target_env = "musl")) {
            Some(stackaddr - guardsize..stackaddr)
        } else if cfg!(all(target_os = "linux", any(target_env = "gnu", target_env = "uclibc"))) {
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
        // SAFETY:
        // `attr` was initialized either by `pthread_attr_init` (FreeBSD) or
        // by `pthread_attr_get_np`, and is not used after this point.
        assert_eq!(unsafe { libc::pthread_attr_destroy(attr.get()) }, 0);
    }
    ret
}
