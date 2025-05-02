//! Provides the [`assert_unsafe_precondition`] macro as well as some utility functions that cover
//! common preconditions.

use crate::intrinsics::{self, const_eval_select};

/// Checks that the preconditions of an unsafe function are followed.
///
/// The check is enabled at runtime if debug assertions are enabled when the
/// caller is monomorphized. In const-eval/Miri checks implemented with this
/// macro for language UB are always ignored.
///
/// This macro should be called as
/// `assert_unsafe_precondition!(check_{library,language}_ub, "message", (ident: type = expr, ident: type = expr) => check_expr)`
/// where each `expr` will be evaluated and passed in as function argument `ident: type`. Then all
/// those arguments are passed to a function with the body `check_expr`.
/// Pick `check_language_ub` when this is guarding a violation of language UB, i.e., immediate UB
/// according to the Rust Abstract Machine. Pick `check_library_ub` when this is guarding a violation
/// of a documented library precondition that does not *immediately* lead to language UB.
///
/// If `check_library_ub` is used but the check is actually guarding language UB, the check will
/// slow down const-eval/Miri and we'll get the panic message instead of the interpreter's nice
/// diagnostic, but our ability to detect UB is unchanged.
/// But if `check_language_ub` is used when the check is actually for library UB, the check is
/// omitted in const-eval/Miri and thus if we eventually execute language UB which relies on the
/// library UB, the backtrace Miri reports may be far removed from original cause.
///
/// These checks are behind a condition which is evaluated at codegen time, not expansion time like
/// [`debug_assert`]. This means that a standard library built with optimizations and debug
/// assertions disabled will have these checks optimized out of its monomorphizations, but if a
/// caller of the standard library has debug assertions enabled and monomorphizes an expansion of
/// this macro, that monomorphization will contain the check.
///
/// Since these checks cannot be optimized out in MIR, some care must be taken in both call and
/// implementation to mitigate their compile-time overhead. Calls to this macro always expand to
/// this structure:
/// ```ignore (pseudocode)
/// if ::core::intrinsics::check_language_ub() {
///     precondition_check(args)
/// }
/// ```
/// where `precondition_check` is monomorphic with the attributes `#[rustc_nounwind]`, `#[inline]` and
/// `#[rustc_no_mir_inline]`. This combination of attributes ensures that the actual check logic is
/// compiled only once and generates a minimal amount of IR because the check cannot be inlined in
/// MIR, but *can* be inlined and fully optimized by a codegen backend.
///
/// Callers should avoid introducing any other `let` bindings or any code outside this macro in
/// order to call it. Since the precompiled standard library is built with full debuginfo and these
/// variables cannot be optimized out in MIR, an innocent-looking `let` can produce enough
/// debuginfo to have a measurable compile-time impact on debug builds.
#[macro_export]
#[unstable(feature = "ub_checks", issue = "none")]
macro_rules! assert_unsafe_precondition {
    ($kind:ident, $message:expr, ($($name:ident:$ty:ty = $arg:expr),*$(,)?) => $e:expr $(,)?) => {
        {
            // This check is inlineable, but not by the MIR inliner.
            // The reason for this is that the MIR inliner is in an exceptionally bad position
            // to think about whether or not to inline this. In MIR, this call is gated behind `debug_assertions`,
            // which will codegen to `false` in release builds. Inlining the check would be wasted work in that case and
            // would be bad for compile times.
            //
            // LLVM on the other hand sees the constant branch, so if it's `false`, it can immediately delete it without
            // inlining the check. If it's `true`, it can inline it and get significantly better performance.
            #[rustc_no_mir_inline]
            #[inline]
            #[rustc_nounwind]
            #[track_caller]
            const fn precondition_check($($name:$ty),*) {
                if !$e {
                    let msg = concat!("unsafe precondition(s) violated: ", $message,
                        "\n\nThis indicates a bug in the program. \
                        This Undefined Behavior check is optional, and cannot be relied on for safety.");
                    ::core::panicking::panic_nounwind_fmt(::core::fmt::Arguments::new_const(&[msg]), false);
                }
            }

            if ::core::ub_checks::$kind() {
                precondition_check($($arg,)*);
            }
        }
    };
}
#[unstable(feature = "ub_checks", issue = "none")]
pub use assert_unsafe_precondition;
/// Checking library UB is always enabled when UB-checking is done
/// (and we use a reexport so that there is no unnecessary wrapper function).
#[unstable(feature = "ub_checks", issue = "none")]
pub use intrinsics::ub_checks as check_library_ub;

/// Determines whether we should check for language UB.
///
/// The intention is to not do that when running in the interpreter, as that one has its own
/// language UB checks which generally produce better errors.
#[inline]
#[rustc_allow_const_fn_unstable(const_eval_select)]
pub(crate) const fn check_language_ub() -> bool {
    // Only used for UB checks so we may const_eval_select.
    intrinsics::ub_checks()
        && const_eval_select!(
            @capture { } -> bool:
            if const {
                // Always disable UB checks.
                false
            } else {
                // Disable UB checks in Miri.
                !cfg!(miri)
            }
        )
}

/// Checks whether `ptr` is properly aligned with respect to the given alignment, and
/// if `is_zst == false`, that `ptr` is not null.
///
/// In `const` this is approximate and can fail spuriously. It is primarily intended
/// for `assert_unsafe_precondition!` with `check_language_ub`, in which case the
/// check is anyway not executed in `const`.
#[inline]
#[rustc_allow_const_fn_unstable(const_eval_select)]
pub(crate) const fn maybe_is_aligned_and_not_null(
    ptr: *const (),
    align: usize,
    is_zst: bool,
) -> bool {
    // This is just for safety checks so we can const_eval_select.
    const_eval_select!(
        @capture { ptr: *const (), align: usize, is_zst: bool } -> bool:
        if const {
            is_zst || !ptr.is_null()
        } else {
            ptr.is_aligned_to(align) && (is_zst || !ptr.is_null())
        }
    )
}

#[inline]
pub(crate) const fn is_valid_allocation_size(size: usize, len: usize) -> bool {
    let max_len = if size == 0 { usize::MAX } else { isize::MAX as usize / size };
    len <= max_len
}

/// Checks whether the regions of memory starting at `src` and `dst` of size
/// `count * size` do *not* overlap.
///
/// Note that in const-eval this function just returns `true` and therefore must
/// only be used with `assert_unsafe_precondition!`, similar to `is_aligned_and_not_null`.
#[inline]
#[rustc_allow_const_fn_unstable(const_eval_select)]
pub(crate) const fn maybe_is_nonoverlapping(
    src: *const (),
    dst: *const (),
    size: usize,
    count: usize,
) -> bool {
    // This is just for safety checks so we can const_eval_select.
    const_eval_select!(
        @capture { src: *const (), dst: *const (), size: usize, count: usize } -> bool:
        if const {
            true
        } else {
            let src_usize = src.addr();
            let dst_usize = dst.addr();
            let Some(size) = size.checked_mul(count) else {
                crate::panicking::panic_nounwind(
                    "is_nonoverlapping: `size_of::<T>() * count` overflows a usize",
                )
            };
            let diff = src_usize.abs_diff(dst_usize);
            // If the absolute distance between the ptrs is at least as big as the size of the buffer,
            // they do not overlap.
            diff >= size
        }
    )
}
