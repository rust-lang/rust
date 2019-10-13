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

/// Emits a machine instruction hinting to the processor that it is running in busy-wait
/// spin-loop ("spin lock").
///
/// For a discussion of different locking strategies and their trade-offs, see
/// [`core::sync::atomic::spin_loop_hint`].
///
/// **Note**: On platforms that do not support receiving spin-loop hints this function does not
/// do anything at all.
///
/// [`core::sync::atomic::spin_loop_hint`]: ../sync/atomic/fn.spin_loop_hint.html
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

/// An identity function that *__hints__* to the compiler to be maximally pessimistic about what
/// `black_box` could do.
///
/// [`std::convert::identity`]: https://doc.rust-lang.org/core/convert/fn.identity.html
///
/// Unlike [`std::convert::identity`], a Rust compiler is encouraged to assume that `black_box` can
/// use `x` in any possible valid way that Rust code is allowed to without introducing undefined
/// behavior in the calling code. This property makes `black_box` useful for writing code in which
/// certain optimizations are not desired, such as benchmarks.
///
/// Note however, that `black_box` is only (and can only be) provided on a "best-effort" basis. The
/// extent to which it can block optimisations may vary depending upon the platform and code-gen
/// backend used. Programs cannot rely on `black_box` for *correctness* in any way.
#[inline]
#[unstable(feature = "test", issue = "50297")]
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
