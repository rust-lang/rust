//! Signal handler for rustc
//! Primarily used to extract a backtrace from stack overflow

use std::alloc::{alloc, Layout};
use std::{fmt, mem, ptr};

extern "C" {
    fn backtrace_symbols_fd(buffer: *const *mut libc::c_void, size: libc::c_int, fd: libc::c_int);
}

fn backtrace_stderr(buffer: &[*mut libc::c_void]) {
    let size = buffer.len().try_into().unwrap_or_default();
    unsafe { backtrace_symbols_fd(buffer.as_ptr(), size, libc::STDERR_FILENO) };
}

/// Unbuffered, unsynchronized writer to stderr.
///
/// Only acceptable because everything will end soon anyways.
struct RawStderr(());

impl fmt::Write for RawStderr {
    fn write_str(&mut self, s: &str) -> Result<(), fmt::Error> {
        let ret = unsafe { libc::write(libc::STDERR_FILENO, s.as_ptr().cast(), s.len()) };
        if ret == -1 { Err(fmt::Error) } else { Ok(()) }
    }
}

/// We don't really care how many bytes we actually get out. SIGSEGV comes for our head.
/// Splash stderr with letters of our own blood to warn our friends about the monster.
macro raw_errln($tokens:tt) {
    let _ = ::core::fmt::Write::write_fmt(&mut RawStderr(()), format_args!($tokens));
    let _ = ::core::fmt::Write::write_char(&mut RawStderr(()), '\n');
}

/// Signal handler installed for SIGSEGV
extern "C" fn print_stack_trace(_: libc::c_int) {
    const MAX_FRAMES: usize = 256;
    // Reserve data segment so we don't have to malloc in a signal handler, which might fail
    // in incredibly undesirable and unexpected ways due to e.g. the allocator deadlocking
    static mut STACK_TRACE: [*mut libc::c_void; MAX_FRAMES] = [ptr::null_mut(); MAX_FRAMES];
    let stack = unsafe {
        // Collect return addresses
        let depth = libc::backtrace(STACK_TRACE.as_mut_ptr(), MAX_FRAMES as i32);
        if depth == 0 {
            return;
        }
        &STACK_TRACE.as_slice()[0..(depth as _)]
    };

    // Just a stack trace is cryptic. Explain what we're doing.
    raw_errln!("error: rustc interrupted by SIGSEGV, printing backtrace\n");
    let mut written = 1;
    let mut consumed = 0;
    // Begin elaborating return addrs into symbols and writing them directly to stderr
    // Most backtraces are stack overflow, most stack overflows are from recursion
    // Check for cycles before writing 250 lines of the same ~5 symbols
    let cycled = |(runner, walker)| runner == walker;
    let mut cyclic = false;
    if let Some(period) = stack.iter().skip(1).step_by(2).zip(stack).position(cycled) {
        let period = period.saturating_add(1); // avoid "what if wrapped?" branches
        let Some(offset) = stack.iter().skip(period).zip(stack).position(cycled) else {
            // impossible.
            return;
        };

        // Count matching trace slices, else we could miscount "biphasic cycles"
        // with the same period + loop entry but a different inner loop
        let next_cycle = stack[offset..].chunks_exact(period).skip(1);
        let cycles = 1 + next_cycle
            .zip(stack[offset..].chunks_exact(period))
            .filter(|(next, prev)| next == prev)
            .count();
        backtrace_stderr(&stack[..offset]);
        written += offset;
        consumed += offset;
        if cycles > 1 {
            raw_errln!("\n### cycle encountered after {offset} frames with period {period}");
            backtrace_stderr(&stack[consumed..consumed + period]);
            raw_errln!("### recursed {cycles} times\n");
            written += period + 4;
            consumed += period * cycles;
            cyclic = true;
        };
    }
    let rem = &stack[consumed..];
    backtrace_stderr(rem);
    raw_errln!("");
    written += rem.len() + 1;

    let random_depth = || 8 * 16; // chosen by random diceroll (2d20)
    if cyclic || stack.len() > random_depth() {
        // technically speculation, but assert it with confidence anyway.
        // rustc only arrived in this signal handler because bad things happened
        // and this message is for explaining it's not the programmer's fault
        raw_errln!("note: rustc unexpectedly overflowed its stack! this is a bug");
        written += 1;
    }
    if stack.len() == MAX_FRAMES {
        raw_errln!("note: maximum backtrace depth reached, frames may have been lost");
        written += 1;
    }
    raw_errln!("note: we would appreciate a report at https://github.com/rust-lang/rust");
    written += 1;
    if written > 24 {
        // We probably just scrolled the earlier "we got SIGSEGV" message off the terminal
        raw_errln!("note: backtrace dumped due to SIGSEGV! resuming signal");
    };
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
