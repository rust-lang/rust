#![stable(feature = "core_hint", since = "1.27.0")]

//! Hints to compiler that affects how code should be emitted or optimized.
//!
//! Hints may be compile time or runtime.

use crate::mem::MaybeUninit;
use crate::{intrinsics, ub_checks};

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
///             unsafe { std::hint::unreachable_unchecked() }
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
/// example, as the compiler is able to prove that a division by zero is not
/// possible, benchmarking reveals that `unreachable_unchecked()` provides
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
    ub_checks::assert_unsafe_precondition!(
        check_language_ub,
        "hint::unreachable_unchecked must never be reached",
        () => false
    );
    // SAFETY: the safety contract for `intrinsics::unreachable` must
    // be upheld by the caller.
    unsafe { intrinsics::unreachable() }
}

/// Makes a *soundness* promise to the compiler that `cond` holds.
///
/// This may allow the optimizer to simplify things, but it might also make the generated code
/// slower. Either way, calling it will most likely make compilation take longer.
///
/// You may know this from other places as
/// [`llvm.assume`](https://llvm.org/docs/LangRef.html#llvm-assume-intrinsic) or, in C,
/// [`__builtin_assume`](https://clang.llvm.org/docs/LanguageExtensions.html#builtin-assume).
///
/// This promotes a correctness requirement to a soundness requirement. Don't do that without
/// very good reason.
///
/// # Usage
///
/// This is a situational tool for micro-optimization, and is allowed to do nothing. Any use
/// should come with a repeatable benchmark to show the value, with the expectation to drop it
/// later should the optimizer get smarter and no longer need it.
///
/// The more complicated the condition, the less likely this is to be useful. For example,
/// `assert_unchecked(foo.is_sorted())` is a complex enough value that the compiler is unlikely
/// to be able to take advantage of it.
///
/// There's also no need to `assert_unchecked` basic properties of things.  For example, the
/// compiler already knows the range of `count_ones`, so there is no benefit to
/// `let n = u32::count_ones(x); assert_unchecked(n <= u32::BITS);`.
///
/// `assert_unchecked` is logically equivalent to `if !cond { unreachable_unchecked(); }`. If
/// ever you are tempted to write `assert_unchecked(false)`, you should instead use
/// [`unreachable_unchecked()`] directly.
///
/// # Safety
///
/// `cond` must be `true`. It is immediate UB to call this with `false`.
///
/// # Example
///
/// ```
/// use core::hint;
///
/// /// # Safety
/// ///
/// /// `p` must be nonnull and valid
/// pub unsafe fn next_value(p: *const i32) -> i32 {
///     // SAFETY: caller invariants guarantee that `p` is not null
///     unsafe { hint::assert_unchecked(!p.is_null()) }
///
///     if p.is_null() {
///         return -1;
///     } else {
///         // SAFETY: caller invariants guarantee that `p` is valid
///         unsafe { *p + 1 }
///     }
/// }
/// ```
///
/// Without the `assert_unchecked`, the above function produces the following with optimizations
/// enabled:
///
/// ```asm
/// next_value:
///         test    rdi, rdi
///         je      .LBB0_1
///         mov     eax, dword ptr [rdi]
///         inc     eax
///         ret
/// .LBB0_1:
///         mov     eax, -1
///         ret
/// ```
///
/// Adding the assertion allows the optimizer to remove the extra check:
///
/// ```asm
/// next_value:
///         mov     eax, dword ptr [rdi]
///         inc     eax
///         ret
/// ```
///
/// This example is quite unlike anything that would be used in the real world: it is redundant
/// to put an assertion right next to code that checks the same thing, and dereferencing a
/// pointer already has the builtin assumption that it is nonnull. However, it illustrates the
/// kind of changes the optimizer can make even when the behavior is less obviously related.
#[track_caller]
#[inline(always)]
#[doc(alias = "assume")]
#[stable(feature = "hint_assert_unchecked", since = "1.81.0")]
#[rustc_const_stable(feature = "hint_assert_unchecked", since = "1.81.0")]
pub const unsafe fn assert_unchecked(cond: bool) {
    // SAFETY: The caller promised `cond` is true.
    unsafe {
        ub_checks::assert_unsafe_precondition!(
            check_language_ub,
            "hint::assert_unchecked must never be called when the condition is false",
            (cond: bool = cond) => cond,
        );
        crate::intrinsics::assume(cond);
    }
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
/// ```ignore-wasm
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
#[inline(always)]
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

    #[cfg(target_arch = "riscv32")]
    {
        crate::arch::riscv32::pause();
    }

    #[cfg(target_arch = "riscv64")]
    {
        crate::arch::riscv64::pause();
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
    {
        // SAFETY: the `cfg` attr ensures that we only execute this on aarch64 targets.
        unsafe { crate::arch::aarch64::__isb(crate::arch::aarch64::SY) };
    }

    #[cfg(all(target_arch = "arm", target_feature = "v6"))]
    {
        // SAFETY: the `cfg` attr ensures that we only execute this on arm targets
        // with support for the v6 feature.
        unsafe { crate::arch::arm::__yield() };
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
/// <div class="warning">
///
/// Note however, that `black_box` is only (and can only be) provided on a "best-effort" basis. The
/// extent to which it can block optimisations may vary depending upon the platform and code-gen
/// backend used. Programs cannot rely on `black_box` for *correctness*, beyond it behaving as the
/// identity function. As such, it **must not be relied upon to control critical program behavior.**
/// This also means that this function does not offer any guarantees for cryptographic or security
/// purposes.
///
/// This limitation is not specific to `black_box`; there is no mechanism in the entire Rust
/// language that can provide the guarantees required for constant-time cryptography.
/// (There is also no such mechanism in LLVM, so the same is true for every other LLVM-based compiler.)
///
/// </div>
///
/// [`std::convert::identity`]: crate::convert::identity
///
/// # When is this useful?
///
/// While not suitable in those mission-critical cases, `black_box`'s functionality can generally be
/// relied upon for benchmarking, and should be used there. It will try to ensure that the
/// compiler doesn't optimize away part of the intended test code based on context. For
/// example:
///
/// ```
/// fn contains(haystack: &[&str], needle: &str) -> bool {
///     haystack.iter().any(|x| x == &needle)
/// }
///
/// pub fn benchmark() {
///     let haystack = vec!["abc", "def", "ghi", "jkl", "mno"];
///     let needle = "ghi";
///     for _ in 0..10 {
///         contains(&haystack, needle);
///     }
/// }
/// ```
///
/// The compiler could theoretically make optimizations like the following:
///
/// - The `needle` and `haystack` do not change, move the call to `contains` outside the loop and
///   delete the loop
/// - Inline `contains`
/// - `needle` and `haystack` have values known at compile time, `contains` is always true. Remove
///   the call and replace with `true`
/// - Nothing is done with the result of `contains`: delete this function call entirely
/// - `benchmark` now has no purpose: delete this function
///
/// It is not likely that all of the above happens, but the compiler is definitely able to make some
/// optimizations that could result in a very inaccurate benchmark. This is where `black_box` comes
/// in:
///
/// ```
/// use std::hint::black_box;
///
/// // Same `contains` function.
/// fn contains(haystack: &[&str], needle: &str) -> bool {
///     haystack.iter().any(|x| x == &needle)
/// }
///
/// pub fn benchmark() {
///     let haystack = vec!["abc", "def", "ghi", "jkl", "mno"];
///     let needle = "ghi";
///     for _ in 0..10 {
///         // Force the compiler to run `contains`, even though it is a pure function whose
///         // results are unused.
///         black_box(contains(
///             // Prevent the compiler from making assumptions about the input.
///             black_box(&haystack),
///             black_box(needle),
///         ));
///     }
/// }
/// ```
///
/// This essentially tells the compiler to block optimizations across any calls to `black_box`. So,
/// it now:
///
/// - Treats both arguments to `contains` as unpredictable: the body of `contains` can no longer be
///   optimized based on argument values
/// - Treats the call to `contains` and its result as volatile: the body of `benchmark` cannot
///   optimize this away
///
/// This makes our benchmark much more realistic to how the function would actually be used, where
/// arguments are usually not known at compile time and the result is used in some way.
///
/// # How to use this
///
/// In practice, `black_box` serves two purposes:
///
/// 1. It prevents the compiler from making optimizations related to the value returned by `black_box`
/// 2. It forces the value passed to `black_box` to be calculated, even if the return value of `black_box` is unused
///
/// ```
/// use std::hint::black_box;
///
/// let zero = 0;
/// let five = 5;
///
/// // The compiler will see this and remove the `* five` call, because it knows that multiplying
/// // any integer by 0 will result in 0.
/// let c = zero * five;
///
/// // Adding `black_box` here disables the compiler's ability to reason about the first operand in the multiplication.
/// // It is forced to assume that it can be any possible number, so it cannot remove the `* five`
/// // operation.
/// let c = black_box(zero) * five;
/// ```
///
/// While most cases will not be as clear-cut as the above example, it still illustrates how
/// `black_box` can be used. When benchmarking a function, you usually want to wrap its inputs in
/// `black_box` so the compiler cannot make optimizations that would be unrealistic in real-life
/// use.
///
/// ```
/// use std::hint::black_box;
///
/// // This is a simple function that increments its input by 1. Note that it is pure, meaning it
/// // has no side-effects. This function has no effect if its result is unused. (An example of a
/// // function *with* side-effects is `println!()`.)
/// fn increment(x: u8) -> u8 {
///     x + 1
/// }
///
/// // Here, we call `increment` but discard its result. The compiler, seeing this and knowing that
/// // `increment` is pure, will eliminate this function call entirely. This may not be desired,
/// // though, especially if we're trying to track how much time `increment` takes to execute.
/// let _ = increment(black_box(5));
///
/// // Here, we force `increment` to be executed. This is because the compiler treats `black_box`
/// // as if it has side-effects, and thus must compute its input.
/// let _ = black_box(increment(black_box(5)));
/// ```
///
/// There may be additional situations where you want to wrap the result of a function in
/// `black_box` to force its execution. This is situational though, and may not have any effect
/// (such as when the function returns a zero-sized type such as [`()` unit][unit]).
///
/// Note that `black_box` has no effect on how its input is treated, only its output. As such,
/// expressions passed to `black_box` may still be optimized:
///
/// ```
/// use std::hint::black_box;
///
/// // The compiler sees this...
/// let y = black_box(5 * 10);
///
/// // ...as this. As such, it will likely simplify `5 * 10` to just `50`.
/// let _0 = 5 * 10;
/// let y = black_box(_0);
/// ```
///
/// In the above example, the `5 * 10` expression is considered distinct from the `black_box` call,
/// and thus is still optimized by the compiler. You can prevent this by moving the multiplication
/// operation outside of `black_box`:
///
/// ```
/// use std::hint::black_box;
///
/// // No assumptions can be made about either operand, so the multiplication is not optimized out.
/// let y = black_box(5) * black_box(10);
/// ```
///
/// During constant evaluation, `black_box` is treated as a no-op.
#[inline]
#[stable(feature = "bench_black_box", since = "1.66.0")]
#[rustc_const_stable(feature = "const_black_box", since = "1.86.0")]
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
#[must_use] // <-- :)
#[inline(always)]
pub const fn must_use<T>(value: T) -> T {
    value
}

/// Hints to the compiler that a branch condition is likely to be true.
/// Returns the value passed to it.
///
/// It can be used with `if` or boolean `match` expressions.
///
/// When used outside of a branch condition, it may still influence a nearby branch, but
/// probably will not have any effect.
///
/// It can also be applied to parts of expressions, such as `likely(a) && unlikely(b)`, or to
/// compound expressions, such as `likely(a && b)`. When applied to compound expressions, it has
/// the following effect:
/// ```text
///     likely(!a) => !unlikely(a)
///     likely(a && b) => likely(a) && likely(b)
///     likely(a || b) => a || likely(b)
/// ```
///
/// See also the function [`cold_path()`] which may be more appropriate for idiomatic Rust code.
///
/// # Examples
///
/// ```
/// #![feature(likely_unlikely)]
/// use core::hint::likely;
///
/// fn foo(x: i32) {
///     if likely(x > 0) {
///         println!("this branch is likely to be taken");
///     } else {
///         println!("this branch is unlikely to be taken");
///     }
///
///     match likely(x > 0) {
///         true => println!("this branch is likely to be taken"),
///         false => println!("this branch is unlikely to be taken"),
///     }
///
///     // Use outside of a branch condition may still influence a nearby branch
///     let cond = likely(x != 0);
///     if cond {
///         println!("this branch is likely to be taken");
///     }
/// }
/// ```
///
///
#[unstable(feature = "likely_unlikely", issue = "136873")]
#[inline(always)]
pub const fn likely(b: bool) -> bool {
    crate::intrinsics::likely(b)
}

/// Hints to the compiler that a branch condition is unlikely to be true.
/// Returns the value passed to it.
///
/// It can be used with `if` or boolean `match` expressions.
///
/// When used outside of a branch condition, it may still influence a nearby branch, but
/// probably will not have any effect.
///
/// It can also be applied to parts of expressions, such as `likely(a) && unlikely(b)`, or to
/// compound expressions, such as `unlikely(a && b)`. When applied to compound expressions, it has
/// the following effect:
/// ```text
///     unlikely(!a) => !likely(a)
///     unlikely(a && b) => a && unlikely(b)
///     unlikely(a || b) => unlikely(a) || unlikely(b)
/// ```
///
/// See also the function [`cold_path()`] which may be more appropriate for idiomatic Rust code.
///
/// # Examples
///
/// ```
/// #![feature(likely_unlikely)]
/// use core::hint::unlikely;
///
/// fn foo(x: i32) {
///     if unlikely(x > 0) {
///         println!("this branch is unlikely to be taken");
///     } else {
///         println!("this branch is likely to be taken");
///     }
///
///     match unlikely(x > 0) {
///         true => println!("this branch is unlikely to be taken"),
///         false => println!("this branch is likely to be taken"),
///     }
///
///     // Use outside of a branch condition may still influence a nearby branch
///     let cond = unlikely(x != 0);
///     if cond {
///         println!("this branch is likely to be taken");
///     }
/// }
/// ```
#[unstable(feature = "likely_unlikely", issue = "136873")]
#[inline(always)]
pub const fn unlikely(b: bool) -> bool {
    crate::intrinsics::unlikely(b)
}

/// Hints to the compiler that given path is cold, i.e., unlikely to be taken. The compiler may
/// choose to optimize paths that are not cold at the expense of paths that are cold.
///
/// # Examples
///
/// ```
/// #![feature(cold_path)]
/// use core::hint::cold_path;
///
/// fn foo(x: &[i32]) {
///     if let Some(first) = x.get(0) {
///         // this is the fast path
///     } else {
///         // this path is unlikely
///         cold_path();
///     }
/// }
///
/// fn bar(x: i32) -> i32 {
///     match x {
///         1 => 10,
///         2 => 100,
///         3 => { cold_path(); 1000 }, // this branch is unlikely
///         _ => { cold_path(); 10000 }, // this is also unlikely
///     }
/// }
/// ```
#[unstable(feature = "cold_path", issue = "136873")]
#[inline(always)]
pub const fn cold_path() {
    crate::intrinsics::cold_path()
}

/// Returns either `true_val` or `false_val` depending on the value of
/// `condition`, with a hint to the compiler that `condition` is unlikely to be
/// correctly predicted by a CPU’s branch predictor.
///
/// This method is functionally equivalent to
/// ```ignore (this is just for illustrative purposes)
/// fn select_unpredictable<T>(b: bool, true_val: T, false_val: T) -> T {
///     if b { true_val } else { false_val }
/// }
/// ```
/// but might generate different assembly. In particular, on platforms with
/// a conditional move or select instruction (like `cmov` on x86 or `csel`
/// on ARM) the optimizer might use these instructions to avoid branches,
/// which can benefit performance if the branch predictor is struggling
/// with predicting `condition`, such as in an implementation of binary
/// search.
///
/// Note however that this lowering is not guaranteed (on any platform) and
/// should not be relied upon when trying to write cryptographic constant-time
/// code. Also be aware that this lowering might *decrease* performance if
/// `condition` is well-predictable. It is advisable to perform benchmarks to
/// tell if this function is useful.
///
/// # Examples
///
/// Distribute values evenly between two buckets:
/// ```
/// use std::hash::BuildHasher;
/// use std::hint;
///
/// fn append<H: BuildHasher>(hasher: &H, v: i32, bucket_one: &mut Vec<i32>, bucket_two: &mut Vec<i32>) {
///     let hash = hasher.hash_one(&v);
///     let bucket = hint::select_unpredictable(hash % 2 == 0, bucket_one, bucket_two);
///     bucket.push(v);
/// }
/// # let hasher = std::collections::hash_map::RandomState::new();
/// # let mut bucket_one = Vec::new();
/// # let mut bucket_two = Vec::new();
/// # append(&hasher, 42, &mut bucket_one, &mut bucket_two);
/// # assert_eq!(bucket_one.len() + bucket_two.len(), 1);
/// ```
#[inline(always)]
#[stable(feature = "select_unpredictable", since = "1.88.0")]
pub fn select_unpredictable<T>(condition: bool, true_val: T, false_val: T) -> T {
    // FIXME(https://github.com/rust-lang/unsafe-code-guidelines/issues/245):
    // Change this to use ManuallyDrop instead.
    let mut true_val = MaybeUninit::new(true_val);
    let mut false_val = MaybeUninit::new(false_val);
    // SAFETY: The value that is not selected is dropped, and the selected one
    // is returned. This is necessary because the intrinsic doesn't drop the
    // value that is  not selected.
    unsafe {
        crate::intrinsics::select_unpredictable(!condition, &mut true_val, &mut false_val)
            .assume_init_drop();
        crate::intrinsics::select_unpredictable(condition, true_val, false_val).assume_init()
    }
}
