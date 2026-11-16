//@ignore-target: windows # only very limited libc on Windows
//@compile-flags: -Zmiri-disable-isolation

use std::mem::transmute;

/// Ensure errno can be written and read.
fn test_errno() {
    #[cfg(any(target_os = "illumos", target_os = "solaris"))]
    use libc::___errno as __errno_location;
    #[cfg(any(target_os = "android", target_os = "netbsd"))]
    use libc::__errno as __errno_location;
    #[cfg(target_os = "linux")]
    use libc::__errno_location;
    #[cfg(any(target_os = "freebsd", target_os = "macos"))]
    use libc::__error as __errno_location;

    unsafe {
        *__errno_location() = 0xBEEF;
        assert_eq!(*__errno_location(), 0xBEEF);
    }
}

fn test_environ() {
    // Just a smoke test for now, checking that the extern static exists.
    extern "C" {
        static mut environ: *const *const libc::c_char;
    }

    unsafe {
        let mut e = environ;
        // Iterate to the end.
        while !(*e).is_null() {
            e = e.add(1);
        }
    }
}

#[cfg(target_os = "linux")]
fn test_sigrt() {
    let min = libc::SIGRTMIN();
    let max = libc::SIGRTMAX();

    // "The Linux kernel supports a range of 33 different real-time
    // signals, numbered 32 to 64"
    assert!(min >= 32);
    assert!(max >= 32);
    assert!(min <= 64);
    assert!(max <= 64);

    // "POSIX.1-2001 requires that an implementation support at least
    // _POSIX_RTSIG_MAX (8) real-time signals."
    assert!(min < max);
    assert!(max - min >= 8)
}

fn test_dlsym() {
    let addr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"notasymbol".as_ptr()) };
    assert!(addr as usize == 0);

    let addr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"strlen".as_ptr()) };
    assert!(addr as usize != 0);
    let strlen: extern "C" fn(*const libc::c_char) -> libc::size_t = unsafe { transmute(addr) };
    assert_eq!(strlen(c"1234".as_ptr()), 4);

    let addr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"environ".as_ptr()) };
    assert!(addr as usize != 0);
    extern "C" {
        static mut environ: *const *const u8;
    }
    assert!(addr as usize == &raw const environ as usize);
}

fn test_getuid() {
    let _val = unsafe { libc::getuid() };
}

fn test_geteuid() {
    let _val = unsafe { libc::geteuid() };
}

fn test_pthread_atfork() {
    extern "C" fn prepare() {}
    extern "C" fn parent() {}
    extern "C" fn child() {}

    // Registering handlers should succeed.
    assert_eq!(unsafe { libc::pthread_atfork(Some(prepare), Some(parent), Some(child)) }, 0);
}

/// Ensure the stack address and size of the current thread can be queried.
#[cfg(target_os = "macos")]
fn test_pthread_get_stack_np() {
    let thread = unsafe { libc::pthread_self() };
    let stack_addr = unsafe { libc::pthread_get_stackaddr_np(thread) };
    let stack_size = unsafe { libc::pthread_get_stacksize_np(thread) };

    assert!(!stack_addr.is_null());
    assert!(stack_size > 0);
    // The stack grows downwards from `stack_addr`, so the bottom of the stack
    // (`stack_addr - stack_size`) must not go below zero.
    assert!(stack_addr.addr() >= stack_size);
}

/// Ensure the stack bounds of the current thread can be queried.
#[cfg(any(target_os = "illumos", target_os = "solaris"))]
fn test_stack_getbounds() {
    let mut stack = std::mem::MaybeUninit::<libc::stack_t>::uninit();
    // Querying the stack bounds should succeed.
    assert_eq!(unsafe { libc::stack_getbounds(stack.as_mut_ptr()) }, 0);

    let stack = unsafe { stack.assume_init() };
    assert!(!stack.ss_sp.is_null());
    assert!(stack.ss_size > 0);
    // We are not running on an alternate signal stack.
    assert_eq!(stack.ss_flags, 0);
}

fn main() {
    test_errno();
    test_environ();
    test_dlsym();
    test_getuid();
    test_geteuid();
    test_pthread_atfork();

    #[cfg(target_os = "macos")]
    test_pthread_get_stack_np();

    #[cfg(any(target_os = "illumos", target_os = "solaris"))]
    test_stack_getbounds();

    #[cfg(target_os = "linux")]
    test_sigrt();
}
