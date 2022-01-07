#![stable(feature = "core_hint", since = "1.27.0")]

//! Hints to compiler that affects how code should be emitted or optimized.
//! Hints may be compile time or runtime.

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
/// assert_eq!(div_1(11, u32::MAX), 0);
/// ```
#[inline]
#[stable(feature = "unreachable", since = "1.27.0")]
#[rustc_const_stable(feature = "const_unreachable_unchecked", since = "1.57.0")]
pub const unsafe fn unreachable_unchecked() -> ! {
    // SAFETY: the safety contract for `intrinsics::unreachable` must
    // be upheld by the caller.
    unsafe { intrinsics::unreachable() }
}

/// Emits a machine instruction to signal the processor that it is running in
/// a busy-wait spin-loop ("spin lock").
///
/// Upon receiving the spin-loop signal the processor can optimize its behavior by,
/// for example, saving power or switching hyper-threads.
///
/// This function is different from [`thread::yield_now`] which directly
/// yields to the system's scheduler, whereas `spin_loop` does not interact
/// with the operating system.
///
/// A common use case for `spin_loop` is implementing bounded optimistic
/// spinning in a CAS loop in synchronization primitives. To avoid problems
/// like priority inversion, it is strongly recommended that the spin loop is
/// terminated after a finite amount of iterations and an appropriate blocking
/// syscall is made.
///
/// **Note**: On platforms that do not support receiving spin-loop hints this
/// function does not do anything at all.
///
/// # Examples
///
/// ```
/// use std::sync::atomic::{AtomicBool, Ordering};
/// use std::sync::Arc;
/// use std::{hint, thread};
///
/// // A shared atomic value that threads will use to coordinate
/// let live = Arc::new(AtomicBool::new(false));
///
/// // In a background thread we'll eventually set the value
/// let bg_work = {
///     let live = live.clone();
///     thread::spawn(move || {
///         // Do some work, then make the value live
///         do_some_work();
///         live.store(true, Ordering::Release);
///     })
/// };
///
/// // Back on our current thread, we wait for the value to be set
/// while !live.load(Ordering::Acquire) {
///     // The spin loop is a hint to the CPU that we're waiting, but probably
///     // not for very long
///     hint::spin_loop();
/// }
///
/// // The value is now set
/// # fn do_some_work() {}
/// do_some_work();
/// bg_work.join()?;
/// # Ok::<(), Box<dyn core::any::Any + Send + 'static>>(())
/// ```
///
/// [`thread::yield_now`]: ../../std/thread/fn.yield_now.html
#[inline]
#[stable(feature = "renamed_spin_loop", since = "1.49.0")]
pub fn spin_loop() {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
    {
        #[cfg(target_arch = "x86")]
        {
            // SAFETY: the `cfg` attr ensures that we only execute this on x86 targets.
            unsafe { crate::arch::x86::_mm_pause() };
        }

        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: the `cfg` attr ensures that we only execute this on x86_64 targets.
            unsafe { crate::arch::x86_64::_mm_pause() };
        }
    }

    // RISC-V platform spin loop hint implementation
    {
        // RISC-V RV32 and RV64 share the same PAUSE instruction, but they are located in different
        // modules in `core::arch`.
        // In this case, here we call `pause` function in each core arch module.
        #[cfg(target_arch = "riscv32")]
        {
            crate::arch::riscv32::pause();
        }
        #[cfg(target_arch = "riscv64")]
        {
            crate::arch::riscv64::pause();
        }
    }

    #[cfg(any(target_arch = "aarch64", all(target_arch = "arm", target_feature = "v6")))]
    {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: the `cfg` attr ensures that we only execute this on aarch64 targets.
            unsafe { crate::arch::aarch64::__isb(crate::arch::aarch64::SY) };
        }
        #[cfg(target_arch = "arm")]
        {
            // SAFETY: the `cfg` attr ensures that we only execute this on arm targets
            // with support for the v6 feature.
            unsafe { crate::arch::arm::__yield() };
        }
    }
}

/// An identity function that *__hints__* to the compiler to be maximally pessimistic about what
/// `black_box` could do.
///
/// Unlike [`std::convert::identity`], a Rust compiler is encouraged to assume that `black_box` can
/// use `dummy` in any possible valid way that Rust code is allowed to without introducing undefined
/// behavior in the calling code. This property makes `black_box` useful for writing code in which
/// certain optimizations are not desired, such as benchmarks.
///
/// Note however, that `black_box` is only (and can only be) provided on a "best-effort" basis. The
/// extent to which it can block optimisations may vary depending upon the platform and code-gen
/// backend used. Programs cannot rely on `black_box` for *correctness* in any way.
///
/// [`std::convert::identity`]: crate::convert::identity
#[inline]
#[unstable(feature = "bench_black_box", issue = "64102")]
#[rustc_const_unstable(feature = "const_black_box", issue = "none")]
pub const fn black_box<T>(dummy: T) -> T {
    crate::intrinsics::black_box(dummy)
}
