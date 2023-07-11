//! Signal handler for rustc
//! Primarily used to extract a backtrace from stack overflow

use std::alloc::{alloc, Layout};
use std::{mem, ptr};

extern "C" {
    fn backtrace_symbols_fd(buffer: *const *mut libc::c_void, size: libc::c_int, fd: libc::c_int);
}

/// Signal handler installed for SIGSEGV
extern "C" fn print_stack_trace(_: libc::c_int) {
    const MAX_FRAMES: usize = 256;
    // Reserve data segment so we don't have to malloc in a signal handler, which might fail
    // in incredibly undesirable and unexpected ways due to e.g. the allocator deadlocking
    static mut STACK_TRACE: [*mut libc::c_void; MAX_FRAMES] = [ptr::null_mut(); MAX_FRAMES];
    unsafe {
        // Collect return addresses
        let depth = libc::backtrace(STACK_TRACE.as_mut_ptr(), MAX_FRAMES as i32);
        if depth == 0 {
            return;
        }
        // Just a stack trace is cryptic. Explain what we're doing.
        write_raw_err("error: rustc interrupted by SIGSEGV, printing stack trace:\n\n");
        // Elaborate return addrs into symbols and write them directly to stderr
        backtrace_symbols_fd(STACK_TRACE.as_ptr(), depth, libc::STDERR_FILENO);
        if depth > 22 {
            // We probably just scrolled that "we got SIGSEGV" message off the terminal
            write_raw_err("\nerror: stack trace dumped due to SIGSEGV, possible stack overflow");
        };
        write_raw_err("\nerror: please report a bug to https://github.com/rust-lang/rust\n");
    }
}

/// Write without locking stderr.
///
/// Only acceptable because everything will end soon anyways.
fn write_raw_err(input: &str) {
    // We do not care how many bytes we actually get out. SIGSEGV comes for our head.
    // Splash stderr with letters of our own blood to warn our friends about the monster.
    let _ = unsafe { libc::write(libc::STDERR_FILENO, input.as_ptr().cast(), input.len()) };
}

/// When SIGSEGV is delivered to the process, print a stack trace and then exit.
pub(super) fn install() {
    unsafe {
        let alt_stack_size: usize = min_sigstack_size() + 64 * 1024;
        let mut alt_stack: libc::stack_t = mem::zeroed();
        alt_stack.ss_sp = alloc(Layout::from_size_align(alt_stack_size, 1).unwrap()).cast();
        alt_stack.ss_size = alt_stack_size;
        libc::sigaltstack(&alt_stack, ptr::null_mut());

        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = print_stack_trace as libc::sighandler_t;
        sa.sa_flags = libc::SA_NODEFER | libc::SA_RESETHAND | libc::SA_ONSTACK;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(libc::SIGSEGV, &sa, ptr::null_mut());
    }
}

/// Modern kernels on modern hardware can have dynamic signal stack sizes.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn min_sigstack_size() -> usize {
    const AT_MINSIGSTKSZ: core::ffi::c_ulong = 51;
    let dynamic_sigstksz = unsafe { libc::getauxval(AT_MINSIGSTKSZ) };
    // If getauxval couldn't find the entry, it returns 0,
    // so take the higher of the "constant" and auxval.
    // This transparently supports older kernels which don't provide AT_MINSIGSTKSZ
    libc::MINSIGSTKSZ.max(dynamic_sigstksz as _)
}

/// Not all OS support hardware where this is needed.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn min_sigstack_size() -> usize {
    libc::MINSIGSTKSZ
}
