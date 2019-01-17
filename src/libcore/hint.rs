#![stable(feature = "core_hint", since = "1.27.0")]

//! Hints to compiler that affects how code should be emitted or optimized.

use intrinsics;

/// Informs the compiler that this point in the code is not reachable, enabling
/// further optimizations.
///
/// # Safety
///
/// Reaching this function is completely *undefined behavior* (UB). In
/// particular, the compiler assumes that all UB must never happen, and
/// therefore will eliminate all branches that reach to a call to
/// `unreachable_unchecked()`.
///
/// Like all instances of UB, if this assumption turns out to be wrong, i.e., the
/// `unreachable_unchecked()` call is actually reachable among all possible
/// control flow, the compiler will apply the wrong optimization strategy, and
/// may sometimes even corrupt seemingly unrelated code, causing
/// difficult-to-debug problems.
///
/// Use this function only when you can prove that the code will never call it.
///
/// The [`unreachable!()`] macro is the safe counterpart of this function, which
/// will panic instead when executed.
///
/// [`unreachable!()`]: ../macro.unreachable.html
///
/// # Example
///
/// ```
/// fn div_1(a: u32, b: u32) -> u32 {
///     use std::hint::unreachable_unchecked;
///
///     // `b.saturating_add(1)` is always positive (not zero),
///     // hence `checked_div` will never return None.
///     // Therefore, the else branch is unreachable.
///     a.checked_div(b.saturating_add(1))
///         .unwrap_or_else(|| unsafe { unreachable_unchecked() })
/// }
///
/// assert_eq!(div_1(7, 0), 7);
/// assert_eq!(div_1(9, 1), 4);
/// assert_eq!(div_1(11, std::u32::MAX), 0);
/// ```
#[inline]
#[stable(feature = "unreachable", since = "1.27.0")]
pub unsafe fn unreachable_unchecked() -> ! {
    intrinsics::unreachable()
}

/// Save power or switch hyperthreads in a busy-wait spin-loop.
///
/// This function is deliberately more primitive than
/// [`std::thread::yield_now`](../../std/thread/fn.yield_now.html) and
/// does not directly yield to the system's scheduler.
/// In some cases it might be useful to use a combination of both functions.
/// Careful benchmarking is advised.
///
/// On some platforms this function may not do anything at all.
#[inline]
#[unstable(feature = "renamed_spin_loop", issue = "55002")]
pub fn spin_loop() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        asm!("pause" ::: "memory" : "volatile");
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        asm!("yield" ::: "memory" : "volatile");
    }
}
