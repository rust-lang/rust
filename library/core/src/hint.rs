#![stable(feature = "core_hint", since = "1.27.0")]

//! Hints to compiler that affects how code should be emitted or optimized.
//! Hints may be compile time or runtime.

use crate::intrinsics;

/// Informs the compiler that the site which is calling this function is not
/// reachable, possibly enabling further optimizations.
///
/// # Safety
///
/// Reaching this function is *Undefined Behavior*.
///
/// As the compiler assumes that all forms of Undefined Behavior can never
/// happen, it will eliminate all branches in the surrounding code that it can
/// determine will invariably lead to a call to `unreachable_unchecked()`.
///
/// If the assumptions embedded in using this function turn out to be wrong -
/// that is, if the site which is calling `unreachable_unchecked()` is actually
/// reachable at runtime - the compiler may have generated nonsensical machine
/// instructions for this situation, including in seemingly unrelated code,
/// causing difficult-to-debug problems.
///
/// Use this function sparingly. Consider using the [`unreachable!`] macro,
/// which may prevent some optimizations but will safely panic in case it is
/// actually reached at runtime. Benchmark your code to find out if using
/// `unreachable_unchecked()` comes with a performance benefit.
///
/// # Examples
///
/// `unreachable_unchecked()` can be used in situations where the compiler
/// can't prove invariants that were previously established. Such situations
/// have a higher chance of occurring if those invariants are upheld by
/// external code that the compiler can't analyze.
/// ```
/// fn prepare_inputs(divisors: &mut Vec<u32>) {
///     // Note to future-self when making changes: The invariant established
///     // here is NOT checked in `do_computation()`; if this changes, you HAVE
///     // to change `do_computation()`.
///     divisors.retain(|divisor| *divisor != 0)
/// }
///
/// /// # Safety
/// /// All elements of `divisor` must be non-zero.
/// unsafe fn do_computation(i: u32, divisors: &[u32]) -> u32 {
///     divisors.iter().fold(i, |acc, divisor| {
///         // Convince the compiler that a division by zero can't happen here
///         // and a check is not needed below.
///         if *divisor == 0 {
///             // Safety: `divisor` can't be zero because of `prepare_inputs`,
///             // but the compiler does not know about this. We *promise*
///             // that we always call `prepare_inputs`.
///             std::hint::unreachable_unchecked()
///         }
///         // The compiler would normally introduce a check here that prevents
///         // a division by zero. However, if `divisor` was zero, the branch
///         // above would reach what we explicitly marked as unreachable.
///         // The compiler concludes that `divisor` can't be zero at this point
///         // and removes the - now proven useless - check.
///         acc / divisor
///     })
/// }
///
/// let mut divisors = vec![2, 0, 4];
/// prepare_inputs(&mut divisors);
/// let result = unsafe {
///     // Safety: prepare_inputs() guarantees that divisors is non-zero
///     do_computation(100, &divisors)
/// };
/// assert_eq!(result, 12);
///
/// ```
///
/// While using `unreachable_unchecked()` is perfectly sound in the following
/// example, the compiler is able to prove that a division by zero is not
/// possible. Benchmarking reveals that `unreachable_unchecked()` provides
/// no benefit over using [`unreachable!`], while the latter does not introduce
/// the possibility of Undefined Behavior.
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
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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

/// An identity function that causes an `unused_must_use` warning to be
/// triggered if the given value is not used (returned, stored in a variable,
/// etc) by the caller.
///
/// This is primarily intended for use in macro-generated code, in which a
/// [`#[must_use]` attribute][must_use] either on a type or a function would not
/// be convenient.
///
/// [must_use]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute
///
/// # Example
///
/// ```
/// #![feature(hint_must_use)]
///
/// use core::fmt;
///
/// pub struct Error(/* ... */);
///
/// #[macro_export]
/// macro_rules! make_error {
///     ($($args:expr),*) => {
///         core::hint::must_use({
///             let error = $crate::make_error(core::format_args!($($args),*));
///             error
///         })
///     };
/// }
///
/// // Implementation detail of make_error! macro.
/// #[doc(hidden)]
/// pub fn make_error(args: fmt::Arguments<'_>) -> Error {
///     Error(/* ... */)
/// }
///
/// fn demo() -> Option<Error> {
///     if true {
///         // Oops, meant to write `return Some(make_error!("..."));`
///         Some(make_error!("..."));
///     }
///     None
/// }
/// #
/// # // Make rustdoc not wrap the whole snippet in fn main, so that $crate::make_error works
/// # fn main() {}
/// ```
///
/// In the above example, we'd like an `unused_must_use` lint to apply to the
/// value created by `make_error!`. However, neither `#[must_use]` on a struct
/// nor `#[must_use]` on a function is appropriate here, so the macro expands
/// using `core::hint::must_use` instead.
///
/// - We wouldn't want `#[must_use]` on the `struct Error` because that would
///   make the following unproblematic code trigger a warning:
///
///   ```
///   # struct Error;
///   #
///   fn f(arg: &str) -> Result<(), Error>
///   # { Ok(()) }
///
///   #[test]
///   fn t() {
///       // Assert that `f` returns error if passed an empty string.
///       // A value of type `Error` is unused here but that's not a problem.
///       f("").unwrap_err();
///   }
///   ```
///
/// - Using `#[must_use]` on `fn make_error` can't help because the return value
///   *is* used, as the right-hand side of a `let` statement. The `let`
///   statement looks useless but is in fact necessary for ensuring that
///   temporaries within the `format_args` expansion are not kept alive past the
///   creation of the `Error`, as keeping them alive past that point can cause
///   autotrait issues in async code:
///
///   ```
///   # #![feature(hint_must_use)]
///   #
///   # struct Error;
///   #
///   # macro_rules! make_error {
///   #     ($($args:expr),*) => {
///   #         core::hint::must_use({
///   #             // If `let` isn't used, then `f()` produces a non-Send future.
///   #             let error = make_error(core::format_args!($($args),*));
///   #             error
///   #         })
///   #     };
///   # }
///   #
///   # fn make_error(args: core::fmt::Arguments<'_>) -> Error {
///   #     Error
///   # }
///   #
///   async fn f() {
///       // Using `let` inside the make_error expansion causes temporaries like
///       // `unsync()` to drop at the semicolon of that `let` statement, which
///       // is prior to the await point. They would otherwise stay around until
///       // the semicolon on *this* statement, which is after the await point,
///       // and the enclosing Future would not implement Send.
///       log(make_error!("look: {:p}", unsync())).await;
///   }
///
///   async fn log(error: Error) {/* ... */}
///
///   // Returns something without a Sync impl.
///   fn unsync() -> *const () {
///       0 as *const ()
///   }
///   #
///   # fn test() {
///   #     fn assert_send(_: impl Send) {}
///   #     assert_send(f());
///   # }
///   ```
#[unstable(feature = "hint_must_use", issue = "94745")]
#[rustc_const_unstable(feature = "hint_must_use", issue = "94745")]
#[must_use] // <-- :)
pub const fn must_use<T>(value: T) -> T {
    value
}
