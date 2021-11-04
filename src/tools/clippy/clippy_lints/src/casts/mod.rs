mod cast_lossless;
mod cast_possible_truncation;
mod cast_possible_wrap;
mod cast_precision_loss;
mod cast_ptr_alignment;
mod cast_ref_to_mut;
mod cast_sign_loss;
mod char_lit_as_u8;
mod fn_to_numeric_cast;
mod fn_to_numeric_cast_any;
mod fn_to_numeric_cast_with_truncation;
mod ptr_as_ptr;
mod unnecessary_cast;
mod utils;

use clippy_utils::is_hir_ty_cfg_dependant;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts from any numerical to a float type where
    /// the receiving type cannot store all values from the original type without
    /// rounding errors. This possible rounding is to be expected, so this lint is
    /// `Allow` by default.
    ///
    /// Basically, this warns on casting any integer with 32 or more bits to `f32`
    /// or any 64-bit integer to `f64`.
    ///
    /// ### Why is this bad?
    /// It's not bad at all. But in some applications it can be
    /// helpful to know where precision loss can take place. This lint can help find
    /// those places in the code.
    ///
    /// ### Example
    /// ```rust
    /// let x = u64::MAX;
    /// x as f64;
    /// ```
    pub CAST_PRECISION_LOSS,
    pedantic,
    "casts that cause loss of precision, e.g., `x as f32` where `x: u64`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts from a signed to an unsigned numerical
    /// type. In this case, negative values wrap around to large positive values,
    /// which can be quite surprising in practice. However, as the cast works as
    /// defined, this lint is `Allow` by default.
    ///
    /// ### Why is this bad?
    /// Possibly surprising results. You can activate this lint
    /// as a one-time check to see where numerical wrapping can arise.
    ///
    /// ### Example
    /// ```rust
    /// let y: i8 = -1;
    /// y as u128; // will return 18446744073709551615
    /// ```
    pub CAST_SIGN_LOSS,
    pedantic,
    "casts from signed types to unsigned types, e.g., `x as u32` where `x: i32`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts between numerical types that may
    /// truncate large values. This is expected behavior, so the cast is `Allow` by
    /// default.
    ///
    /// ### Why is this bad?
    /// In some problem domains, it is good practice to avoid
    /// truncation. This lint can be activated to help assess where additional
    /// checks could be beneficial.
    ///
    /// ### Example
    /// ```rust
    /// fn as_u8(x: u64) -> u8 {
    ///     x as u8
    /// }
    /// ```
    pub CAST_POSSIBLE_TRUNCATION,
    pedantic,
    "casts that may cause truncation of the value, e.g., `x as u8` where `x: u32`, or `x as i32` where `x: f32`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts from an unsigned type to a signed type of
    /// the same size. Performing such a cast is a 'no-op' for the compiler,
    /// i.e., nothing is changed at the bit level, and the binary representation of
    /// the value is reinterpreted. This can cause wrapping if the value is too big
    /// for the target signed type. However, the cast works as defined, so this lint
    /// is `Allow` by default.
    ///
    /// ### Why is this bad?
    /// While such a cast is not bad in itself, the results can
    /// be surprising when this is not the intended behavior, as demonstrated by the
    /// example below.
    ///
    /// ### Example
    /// ```rust
    /// u32::MAX as i32; // will yield a value of `-1`
    /// ```
    pub CAST_POSSIBLE_WRAP,
    pedantic,
    "casts that may cause wrapping around the value, e.g., `x as i32` where `x: u32` and `x > i32::MAX`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts between numerical types that may
    /// be replaced by safe conversion functions.
    ///
    /// ### Why is this bad?
    /// Rust's `as` keyword will perform many kinds of
    /// conversions, including silently lossy conversions. Conversion functions such
    /// as `i32::from` will only perform lossless conversions. Using the conversion
    /// functions prevents conversions from turning into silent lossy conversions if
    /// the types of the input expressions ever change, and make it easier for
    /// people reading the code to know that the conversion is lossless.
    ///
    /// ### Example
    /// ```rust
    /// fn as_u64(x: u8) -> u64 {
    ///     x as u64
    /// }
    /// ```
    ///
    /// Using `::from` would look like this:
    ///
    /// ```rust
    /// fn as_u64(x: u8) -> u64 {
    ///     u64::from(x)
    /// }
    /// ```
    pub CAST_LOSSLESS,
    pedantic,
    "casts using `as` that are known to be lossless, e.g., `x as u64` where `x: u8`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts to the same type, casts of int literals to integer types
    /// and casts of float literals to float types.
    ///
    /// ### Why is this bad?
    /// It's just unnecessary.
    ///
    /// ### Example
    /// ```rust
    /// let _ = 2i32 as i32;
    /// let _ = 0.5 as f32;
    /// ```
    ///
    /// Better:
    ///
    /// ```rust
    /// let _ = 2_i32;
    /// let _ = 0.5_f32;
    /// ```
    pub UNNECESSARY_CAST,
    complexity,
    "cast to the same type, e.g., `x as i32` where `x: i32`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts, using `as` or `pointer::cast`,
    /// from a less-strictly-aligned pointer to a more-strictly-aligned pointer
    ///
    /// ### Why is this bad?
    /// Dereferencing the resulting pointer may be undefined
    /// behavior.
    ///
    /// ### Known problems
    /// Using `std::ptr::read_unaligned` and `std::ptr::write_unaligned` or similar
    /// on the resulting pointer is fine. Is over-zealous: Casts with manual alignment checks or casts like
    /// u64-> u8 -> u16 can be fine. Miri is able to do a more in-depth analysis.
    ///
    /// ### Example
    /// ```rust
    /// let _ = (&1u8 as *const u8) as *const u16;
    /// let _ = (&mut 1u8 as *mut u8) as *mut u16;
    ///
    /// (&1u8 as *const u8).cast::<u16>();
    /// (&mut 1u8 as *mut u8).cast::<u16>();
    /// ```
    pub CAST_PTR_ALIGNMENT,
    pedantic,
    "cast from a pointer to a more-strictly-aligned pointer"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts of function pointers to something other than usize
    ///
    /// ### Why is this bad?
    /// Casting a function pointer to anything other than usize/isize is not portable across
    /// architectures, because you end up losing bits if the target type is too small or end up with a
    /// bunch of extra bits that waste space and add more instructions to the final binary than
    /// strictly necessary for the problem
    ///
    /// Casting to isize also doesn't make sense since there are no signed addresses.
    ///
    /// ### Example
    /// ```rust
    /// // Bad
    /// fn fun() -> i32 { 1 }
    /// let a = fun as i64;
    ///
    /// // Good
    /// fn fun2() -> i32 { 1 }
    /// let a = fun2 as usize;
    /// ```
    pub FN_TO_NUMERIC_CAST,
    style,
    "casting a function pointer to a numeric type other than usize"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts of a function pointer to a numeric type not wide enough to
    /// store address.
    ///
    /// ### Why is this bad?
    /// Such a cast discards some bits of the function's address. If this is intended, it would be more
    /// clearly expressed by casting to usize first, then casting the usize to the intended type (with
    /// a comment) to perform the truncation.
    ///
    /// ### Example
    /// ```rust
    /// // Bad
    /// fn fn1() -> i16 {
    ///     1
    /// };
    /// let _ = fn1 as i32;
    ///
    /// // Better: Cast to usize first, then comment with the reason for the truncation
    /// fn fn2() -> i16 {
    ///     1
    /// };
    /// let fn_ptr = fn2 as usize;
    /// let fn_ptr_truncated = fn_ptr as i32;
    /// ```
    pub FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
    style,
    "casting a function pointer to a numeric type not wide enough to store the address"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts of a function pointer to any integer type.
    ///
    /// ### Why is this bad?
    /// Casting a function pointer to an integer can have surprising results and can occur
    /// accidentally if parantheses are omitted from a function call. If you aren't doing anything
    /// low-level with function pointers then you can opt-out of casting functions to integers in
    /// order to avoid mistakes. Alternatively, you can use this lint to audit all uses of function
    /// pointer casts in your code.
    ///
    /// ### Example
    /// ```rust
    /// // Bad: fn1 is cast as `usize`
    /// fn fn1() -> u16 {
    ///     1
    /// };
    /// let _ = fn1 as usize;
    ///
    /// // Good: maybe you intended to call the function?
    /// fn fn2() -> u16 {
    ///     1
    /// };
    /// let _ = fn2() as usize;
    ///
    /// // Good: maybe you intended to cast it to a function type?
    /// fn fn3() -> u16 {
    ///     1
    /// }
    /// let _ = fn3 as fn() -> u16;
    /// ```
    pub FN_TO_NUMERIC_CAST_ANY,
    restriction,
    "casting a function pointer to any integer type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts of `&T` to `&mut T` anywhere in the code.
    ///
    /// ### Why is this bad?
    /// It’s basically guaranteed to be undefined behaviour.
    /// `UnsafeCell` is the only way to obtain aliasable data that is considered
    /// mutable.
    ///
    /// ### Example
    /// ```rust,ignore
    /// fn x(r: &i32) {
    ///     unsafe {
    ///         *(r as *const _ as *mut _) += 1;
    ///     }
    /// }
    /// ```
    ///
    /// Instead consider using interior mutability types.
    ///
    /// ```rust
    /// use std::cell::UnsafeCell;
    ///
    /// fn x(r: &UnsafeCell<i32>) {
    ///     unsafe {
    ///         *r.get() += 1;
    ///     }
    /// }
    /// ```
    pub CAST_REF_TO_MUT,
    correctness,
    "a cast of reference to a mutable pointer"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions where a character literal is cast
    /// to `u8` and suggests using a byte literal instead.
    ///
    /// ### Why is this bad?
    /// In general, casting values to smaller types is
    /// error-prone and should be avoided where possible. In the particular case of
    /// converting a character literal to u8, it is easy to avoid by just using a
    /// byte literal instead. As an added bonus, `b'a'` is even slightly shorter
    /// than `'a' as u8`.
    ///
    /// ### Example
    /// ```rust,ignore
    /// 'x' as u8
    /// ```
    ///
    /// A better version, using the byte literal:
    ///
    /// ```rust,ignore
    /// b'x'
    /// ```
    pub CHAR_LIT_AS_U8,
    complexity,
    "casting a character literal to `u8` truncates"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `as` casts between raw pointers without changing its mutability,
    /// namely `*const T` to `*const U` and `*mut T` to `*mut U`.
    ///
    /// ### Why is this bad?
    /// Though `as` casts between raw pointers is not terrible, `pointer::cast` is safer because
    /// it cannot accidentally change the pointer's mutability nor cast the pointer to other types like `usize`.
    ///
    /// ### Example
    /// ```rust
    /// let ptr: *const u32 = &42_u32;
    /// let mut_ptr: *mut u32 = &mut 42_u32;
    /// let _ = ptr as *const i32;
    /// let _ = mut_ptr as *mut i32;
    /// ```
    /// Use instead:
    /// ```rust
    /// let ptr: *const u32 = &42_u32;
    /// let mut_ptr: *mut u32 = &mut 42_u32;
    /// let _ = ptr.cast::<i32>();
    /// let _ = mut_ptr.cast::<i32>();
    /// ```
    pub PTR_AS_PTR,
    pedantic,
    "casting using `as` from and to raw pointers that doesn't change its mutability, where `pointer::cast` could take the place of `as`"
}

pub struct Casts {
    msrv: Option<RustcVersion>,
}

impl Casts {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(Casts => [
    CAST_PRECISION_LOSS,
    CAST_SIGN_LOSS,
    CAST_POSSIBLE_TRUNCATION,
    CAST_POSSIBLE_WRAP,
    CAST_LOSSLESS,
    CAST_REF_TO_MUT,
    CAST_PTR_ALIGNMENT,
    UNNECESSARY_CAST,
    FN_TO_NUMERIC_CAST_ANY,
    FN_TO_NUMERIC_CAST,
    FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
    CHAR_LIT_AS_U8,
    PTR_AS_PTR,
]);

impl<'tcx> LateLintPass<'tcx> for Casts {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        if let ExprKind::Cast(cast_expr, cast_to) = expr.kind {
            if is_hir_ty_cfg_dependant(cx, cast_to) {
                return;
            }
            let (cast_from, cast_to) = (
                cx.typeck_results().expr_ty(cast_expr),
                cx.typeck_results().expr_ty(expr),
            );

            if unnecessary_cast::check(cx, expr, cast_expr, cast_from, cast_to) {
                return;
            }

            fn_to_numeric_cast_any::check(cx, expr, cast_expr, cast_from, cast_to);
            fn_to_numeric_cast::check(cx, expr, cast_expr, cast_from, cast_to);
            fn_to_numeric_cast_with_truncation::check(cx, expr, cast_expr, cast_from, cast_to);
            if cast_from.is_numeric() && cast_to.is_numeric() && !in_external_macro(cx.sess(), expr.span) {
                cast_possible_truncation::check(cx, expr, cast_expr, cast_from, cast_to);
                cast_possible_wrap::check(cx, expr, cast_from, cast_to);
                cast_precision_loss::check(cx, expr, cast_from, cast_to);
                cast_lossless::check(cx, expr, cast_expr, cast_from, cast_to);
                cast_sign_loss::check(cx, expr, cast_expr, cast_from, cast_to);
            }
        }

        cast_ref_to_mut::check(cx, expr);
        cast_ptr_alignment::check(cx, expr);
        char_lit_as_u8::check(cx, expr);
        ptr_as_ptr::check(cx, expr, &self.msrv);
    }

    extract_msrv_attr!(LateContext);
}
