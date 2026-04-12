//! ThingOS threading implementation.
//!
//! Threads are created via the `SYS_THREAD_SPAWN` system call and managed
//! through opaque thread handles (u64).  TLS follows ELF Variant II.

use crate::ffi::CStr;
use crate::io;
use crate::num::NonZeroUsize;
use crate::sys::pal::common::{
    SYS_CPUS, SYS_GETTID, SYS_SET_THREAD_NAME, SYS_THREAD_JOIN, SYS_THREAD_SLEEP_NS,
    SYS_THREAD_SPAWN, SYS_THREAD_YIELD, cvt, raw_syscall6, syscall0, syscall1, syscall2,
};
use crate::thread::ThreadInit;
use crate::time::Duration;

/// Minimum stack size for a ThingOS thread (256 KiB).
pub const DEFAULT_MIN_STACK_SIZE: usize = 256 * 1024;

/// A handle to a ThingOS thread.
pub struct Thread {
    /// The kernel-assigned thread handle.
    handle: u64,
}

unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

/// The C-ABI entry point passed to `SYS_THREAD_SPAWN`.
///
/// The kernel calls this with `thread_arg` as the sole argument.
unsafe extern "C" fn thread_entry(thread_arg: u64) {
    // SAFETY: We passed a `Box<ThreadInit>` pointer as the arg.
    let init = unsafe {
        Box::from_raw(core::ptr::with_exposed_provenance_mut::<ThreadInit>(thread_arg as usize))
    };
    let start = init.init();
    start();
}

impl Thread {
    /// Spawn a new thread with at least `stack` bytes of stack.
    ///
    /// # Safety
    /// `init` must be a valid, heap-allocated `ThreadInit` that will be
    /// consumed by the new thread.
    pub unsafe fn new(stack: usize, init: Box<ThreadInit>) -> io::Result<Thread> {
        let thread_arg = Box::into_raw(init).expose_provenance() as u64;
        // SYS_THREAD_SPAWN(entry_fn, stack_size, arg) -> thread_handle
        let handle = cvt(unsafe {
            raw_syscall6(
                SYS_THREAD_SPAWN,
                thread_entry as u64,
                stack as u64,
                thread_arg,
                0,
                0,
                0,
            )
        })? as u64;
        Ok(Thread { handle })
    }

    /// Block until the thread has finished.
    pub fn join(self) {
        unsafe {
            syscall1(SYS_THREAD_JOIN, self.handle);
        }
        // Prevent the destructor from running on an already-joined handle.
        core::mem::forget(self);
    }
}

/// Set the name of the current thread.
pub fn set_name(name: &CStr) {
    let bytes = name.to_bytes();
    unsafe {
        syscall2(SYS_SET_THREAD_NAME, bytes.as_ptr() as u64, bytes.len() as u64);
    }
}

/// Return the numeric OS identifier of the current thread, if available.
pub fn current_os_id() -> Option<u64> {
    let tid = unsafe { syscall0(SYS_GETTID) };
    if tid < 0 { None } else { Some(tid as u64) }
}

/// Return the number of logical CPUs that can run Rust threads in parallel.
///
/// Falls back to 1 if the syscall is unavailable (issue #5 placeholder).
pub fn available_parallelism() -> io::Result<NonZeroUsize> {
    let n = unsafe { syscall0(SYS_CPUS) };
    if n <= 0 {
        // SYS_CPUS not yet implemented in the kernel; return 1 as a safe default.
        Ok(NonZeroUsize::new(1).unwrap())
    } else {
        Ok(NonZeroUsize::new(n as usize).unwrap_or(NonZeroUsize::new(1).unwrap()))
    }
}

/// Voluntarily yield the CPU to the scheduler.
pub fn yield_now() {
    unsafe {
        syscall0(SYS_THREAD_YIELD);
    }
}

/// Sleep for (at least) `dur`.
pub fn sleep(dur: Duration) {
    let ns = dur.as_nanos();
    // Saturate at u64::MAX nanoseconds (~584 years).
    let ns64 = if ns > u64::MAX as u128 { u64::MAX } else { ns as u64 };
    unsafe {
        syscall1(SYS_THREAD_SLEEP_NS, ns64);
    }
}
