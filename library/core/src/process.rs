//! A module for working with processes.
//!
//! Most process-related functionality requires std, but [`abort_immediate`]
//! is available on all targets.

/// Terminates the process in a violent fashion.
///
/// The function will never return and will immediately terminate the current
/// process in a platform specific "abnormal" manner. As a consequence,
/// no destructors on the current stack or any other thread's stack
/// will be run, Rust IO buffers (eg, from `BufWriter`) will not be flushed,
/// and C stdio buffers will not be flushed.
///
/// Unlike [`abort`](../../std/process/fn.abort.html), `abort_immediate` does
/// not attempt to match C `abort()` or otherwise perform a "clean" abort.
/// Instead, it emits code that will crash the process with as little overhead
/// as possible, such as a "halt and catch fire" style instruction. You should
/// generally prefer using `abort` instead except where the absolute minimum
/// overhead is required.
///
/// # Platform-specific behavior
///
/// `abort_immediate` lowers to a trap instruction on *most* architectures; on
/// some architectures it simply lowers to call the unmangled `abort` function.
/// The exact behavior is architecture- and system-dependent.
///
/// On bare-metal (no OS) systems the trap instruction usually causes a
/// *hardware* exception to be raised in a *synchronous* fashion; hardware
/// exceptions have nothing to do with C++ exceptions and are closer in
/// semantics to POSIX signals.
///
/// On hosted applications (applications running under an OS), the trap
/// instruction *usually* terminates the whole process with an exit code that
/// corresponds to `SIGILL` or equivalent, *unless* this signal is handled.
/// Other signals such as `SIGABRT`, `SIGTRAP`, `SIGSEGV`, and `SIGBUS` may be
/// produced instead, depending on specifics. This is not an exhaustive list.
///
/// # Examples
///
/// ```no_run
/// #![feature(abort_immediate)]
///
/// use core::process;
///
/// fn main() {
///     println!("aborting");
///
///     process::abort_immediate();
///
///     // execution never gets here
/// }
/// ```
///
/// The `abort_immediate` function terminates the process, so the destructor
/// will not get run on the example below:
///
/// ```no_run
/// #![feature(abort_immediate)]
///
/// use core::process;
///
/// struct HasDrop;
///
/// impl Drop for HasDrop {
///     fn drop(&mut self) {
///         println!("This will never be printed!");
///     }
/// }
///
/// fn main() {
///     let _x = HasDrop;
///     process::abort_immediate();
///     // the destructor implemented for HasDrop will never get run
/// }
/// ```
///
#[unstable(feature = "abort_immediate", issue = "154601")]
#[rustc_const_unstable(feature = "abort_immediate", issue = "154601")]
#[cold]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
#[doc(alias = "halt")]
pub const fn abort_immediate() -> ! {
    crate::intrinsics::abort()
}
