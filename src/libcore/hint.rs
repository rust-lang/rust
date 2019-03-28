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
pub fn black_box<T>(dummy: T) -> T {
    cfg_if! {
        if #[cfg(any(
            target_arch = "asmjs",
            all(
                target_arch = "wasm32",
                target_os = "emscripten"
            )
        ))] {
            #[inline]
            unsafe fn black_box_impl<T>(d: T) -> T {
                // these targets do not support inline assembly
                let ret = crate::ptr::read_volatile(&d);
                crate::mem::forget(d);
                ret
            }
        } else {
            #[inline]
            unsafe fn black_box_impl<T>(d: T) -> T {
                // we need to "use" the argument in some way LLVM can't
                // introspect.
                asm!("" : : "r"(&d));
                d
            }
        }
    }
    unsafe { black_box_impl(dummy) }
}
