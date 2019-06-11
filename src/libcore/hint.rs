#![stable(feature = "core_hint", since = "1.27.0")]

//! Hints to compiler that affects how code should be emitted or optimized.

use crate::intrinsics;

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
/// Otherwise, consider using the [`unreachable!`] macro, which does not allow
/// optimizations but will panic when executed.
///
/// [`unreachable!`]: ../macro.unreachable.html
///
/// # Example
///
/// ```
/// fn div_1(a: u32, b: u32) -> u32 {
///     use std::hint::unreachable_unchecked;
///
///     // `b.saturating_add(1)` is always positive (not zero),
///     // hence `checked_div` will never return `None`.
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

/// Signals the processor that it is entering a busy-wait spin-loop.
///
/// Upon receiving spin-loop signal the processor can optimize its behavior by, for example, saving
/// power or switching hyper-threads.
///
/// This function is different than [`std::thread::yield_now`] which directly yields to the
/// system's scheduler, whereas `spin_loop` only signals the processor that it is entering a
/// busy-wait spin-loop without yielding control to the system's scheduler.
///
/// Using a busy-wait spin-loop with `spin_loop` is ideally used in situations where a
/// contended lock is held by another thread executed on a different CPU and where the waiting
/// times are relatively small. Because entering busy-wait spin-loop does not trigger the system's
/// scheduler, no overhead for switching threads occurs. However, if the thread holding the
/// contended lock is running on the same CPU, the spin-loop is likely to occupy an entire CPU slice
/// before switching to the thread that holds the lock. If the contending lock is held by a thread
/// on the same CPU or if the waiting times for acquiring the lock are longer, it is often better to
/// use [`std::thread::yield_now`].
///
/// **Note**: On platforms that do not support receiving spin-loop hints this function does not
/// do anything at all.
///
/// [`std::thread::yield_now`]: ../../std/thread/fn.yield_now.html
#[inline]
#[unstable(feature = "renamed_spin_loop", issue = "55002")]
pub fn spin_loop() {
    #[cfg(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        )
    )] {
        #[cfg(target_arch = "x86")] {
            unsafe { crate::arch::x86::_mm_pause() };
        }

        #[cfg(target_arch = "x86_64")] {
            unsafe { crate::arch::x86_64::_mm_pause() };
        }
    }

    #[cfg(
        any(
            target_arch = "aarch64",
            all(target_arch = "arm", target_feature = "v6")
        )
    )] {
        #[cfg(target_arch = "aarch64")] {
            unsafe { crate::arch::aarch64::__yield() };
        }
        #[cfg(target_arch = "arm")] {
            unsafe { crate::arch::arm::__yield() };
        }
    }
}

/// A function that is opaque to the optimizer, to allow benchmarks to
/// pretend to use outputs to assist in avoiding dead-code
/// elimination.
///
/// This function is a no-op, and does not even read from `dummy`.
#[inline]
#[unstable(feature = "test", issue = "27812")]
#[allow(unreachable_code)] // this makes #[cfg] a bit easier below.
pub fn black_box<T>(dummy: T) -> T {
    // We need to "use" the argument in some way LLVM can't introspect, and on
    // targets that support it we can typically leverage inline assembly to do
    // this. LLVM's intepretation of inline assembly is that it's, well, a black
    // box. This isn't the greatest implementation since it probably deoptimizes
    // more than we want, but it's so far good enough.
    #[cfg(not(any(
        target_arch = "asmjs",
        all(
            target_arch = "wasm32",
            target_os = "emscripten"
        )
    )))]
    unsafe {
        asm!("" : : "r"(&dummy));
        return dummy;
    }

    // Not all platforms support inline assembly so try to do something without
    // inline assembly which in theory still hinders at least some optimizations
    // on those targets. This is the "best effort" scenario.
    unsafe {
        let ret = crate::ptr::read_volatile(&dummy);
        crate::mem::forget(dummy);
        ret
    }
}
