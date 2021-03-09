mod cast_lossless;
mod cast_possible_truncation;
mod cast_possible_wrap;
mod cast_precision_loss;
mod cast_ptr_alignment;
mod cast_sign_loss;
mod fn_to_numeric_cast;
mod fn_to_numeric_cast_with_truncation;
mod unnecessary_cast;
mod utils;

use std::borrow::Cow;

use if_chain::if_chain;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, MutTy, Mutability, TyKind, UnOp};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, TypeAndMut, UintTy};
use rustc_semver::RustcVersion;
use rustc_session::{declare_lint_pass, declare_tool_lint, impl_lint_pass};

use crate::utils::sugg::Sugg;
use crate::utils::{
    is_hir_ty_cfg_dependant, meets_msrv, snippet_with_applicability, span_lint, span_lint_and_sugg, span_lint_and_then,
};

declare_clippy_lint! {
    /// **What it does:** Checks for casts from any numerical to a float type where
    /// the receiving type cannot store all values from the original type without
    /// rounding errors. This possible rounding is to be expected, so this lint is
    /// `Allow` by default.
    ///
    /// Basically, this warns on casting any integer with 32 or more bits to `f32`
    /// or any 64-bit integer to `f64`.
    ///
    /// **Why is this bad?** It's not bad at all. But in some applications it can be
    /// helpful to know where precision loss can take place. This lint can help find
    /// those places in the code.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x = u64::MAX;
    /// x as f64;
    /// ```
    pub CAST_PRECISION_LOSS,
    pedantic,
    "casts that cause loss of precision, e.g., `x as f32` where `x: u64`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts from a signed to an unsigned numerical
    /// type. In this case, negative values wrap around to large positive values,
    /// which can be quite surprising in practice. However, as the cast works as
    /// defined, this lint is `Allow` by default.
    ///
    /// **Why is this bad?** Possibly surprising results. You can activate this lint
    /// as a one-time check to see where numerical wrapping can arise.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let y: i8 = -1;
    /// y as u128; // will return 18446744073709551615
    /// ```
    pub CAST_SIGN_LOSS,
    pedantic,
    "casts from signed types to unsigned types, e.g., `x as u32` where `x: i32`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts between numerical types that may
    /// truncate large values. This is expected behavior, so the cast is `Allow` by
    /// default.
    ///
    /// **Why is this bad?** In some problem domains, it is good practice to avoid
    /// truncation. This lint can be activated to help assess where additional
    /// checks could be beneficial.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for casts from an unsigned type to a signed type of
    /// the same size. Performing such a cast is a 'no-op' for the compiler,
    /// i.e., nothing is changed at the bit level, and the binary representation of
    /// the value is reinterpreted. This can cause wrapping if the value is too big
    /// for the target signed type. However, the cast works as defined, so this lint
    /// is `Allow` by default.
    ///
    /// **Why is this bad?** While such a cast is not bad in itself, the results can
    /// be surprising when this is not the intended behavior, as demonstrated by the
    /// example below.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// u32::MAX as i32; // will yield a value of `-1`
    /// ```
    pub CAST_POSSIBLE_WRAP,
    pedantic,
    "casts that may cause wrapping around the value, e.g., `x as i32` where `x: u32` and `x > i32::MAX`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts between numerical types that may
    /// be replaced by safe conversion functions.
    ///
    /// **Why is this bad?** Rust's `as` keyword will perform many kinds of
    /// conversions, including silently lossy conversions. Conversion functions such
    /// as `i32::from` will only perform lossless conversions. Using the conversion
    /// functions prevents conversions from turning into silent lossy conversions if
    /// the types of the input expressions ever change, and make it easier for
    /// people reading the code to know that the conversion is lossless.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for casts to the same type, casts of int literals to integer types
    /// and casts of float literals to float types.
    ///
    /// **Why is this bad?** It's just unnecessary.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for casts, using `as` or `pointer::cast`,
    /// from a less-strictly-aligned pointer to a more-strictly-aligned pointer
    ///
    /// **Why is this bad?** Dereferencing the resulting pointer may be undefined
    /// behavior.
    ///
    /// **Known problems:** Using `std::ptr::read_unaligned` and `std::ptr::write_unaligned` or similar
    /// on the resulting pointer is fine. Is over-zealous: Casts with manual alignment checks or casts like
    /// u64-> u8 -> u16 can be fine. Miri is able to do a more in-depth analysis.
    ///
    /// **Example:**
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
    /// **What it does:** Checks for casts of function pointers to something other than usize
    ///
    /// **Why is this bad?**
    /// Casting a function pointer to anything other than usize/isize is not portable across
    /// architectures, because you end up losing bits if the target type is too small or end up with a
    /// bunch of extra bits that waste space and add more instructions to the final binary than
    /// strictly necessary for the problem
    ///
    /// Casting to isize also doesn't make sense since there are no signed addresses.
    ///
    /// **Example**
    ///
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
    /// **What it does:** Checks for casts of a function pointer to a numeric type not wide enough to
    /// store address.
    ///
    /// **Why is this bad?**
    /// Such a cast discards some bits of the function's address. If this is intended, it would be more
    /// clearly expressed by casting to usize first, then casting the usize to the intended type (with
    /// a comment) to perform the truncation.
    ///
    /// **Example**
    ///
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

declare_lint_pass!(Casts => [
    CAST_PRECISION_LOSS,
    CAST_SIGN_LOSS,
    CAST_POSSIBLE_TRUNCATION,
    CAST_POSSIBLE_WRAP,
    CAST_LOSSLESS,
    UNNECESSARY_CAST,
    CAST_PTR_ALIGNMENT,
    FN_TO_NUMERIC_CAST,
    FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
]);

impl<'tcx> LateLintPass<'tcx> for Casts {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }
        if let ExprKind::Cast(ref cast_expr, cast_to) = expr.kind {
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

            fn_to_numeric_cast::check(cx, expr, cast_expr, cast_from, cast_to);
            fn_to_numeric_cast_with_truncation::check(cx, expr, cast_expr, cast_from, cast_to);
            if cast_from.is_numeric() && cast_to.is_numeric() && !in_external_macro(cx.sess(), expr.span) {
                cast_possible_truncation::check(cx, expr, cast_from, cast_to);
                cast_possible_wrap::check(cx, expr, cast_from, cast_to);
                cast_precision_loss::check(cx, expr, cast_from, cast_to);
                cast_lossless::check(cx, expr, cast_expr, cast_from, cast_to);
                cast_sign_loss::check(cx, expr, cast_expr, cast_from, cast_to);
            }
        }

        cast_ptr_alignment::check(cx, expr);
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for casts of `&T` to `&mut T` anywhere in the code.
    ///
    /// **Why is this bad?** It’s basically guaranteed to be undefined behaviour.
    /// `UnsafeCell` is the only way to obtain aliasable data that is considered
    /// mutable.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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

declare_lint_pass!(RefToMut => [CAST_REF_TO_MUT]);

impl<'tcx> LateLintPass<'tcx> for RefToMut {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Unary(UnOp::Deref, e) = &expr.kind;
            if let ExprKind::Cast(e, t) = &e.kind;
            if let TyKind::Ptr(MutTy { mutbl: Mutability::Mut, .. }) = t.kind;
            if let ExprKind::Cast(e, t) = &e.kind;
            if let TyKind::Ptr(MutTy { mutbl: Mutability::Not, .. }) = t.kind;
            if let ty::Ref(..) = cx.typeck_results().node_type(e.hir_id).kind();
            then {
                span_lint(
                    cx,
                    CAST_REF_TO_MUT,
                    expr.span,
                    "casting `&T` to `&mut T` may cause undefined behavior, consider instead using an `UnsafeCell`",
                );
            }
        }
    }
}

const PTR_AS_PTR_MSRV: RustcVersion = RustcVersion::new(1, 38, 0);

declare_clippy_lint! {
    /// **What it does:** Checks for expressions where a character literal is cast
    /// to `u8` and suggests using a byte literal instead.
    ///
    /// **Why is this bad?** In general, casting values to smaller types is
    /// error-prone and should be avoided where possible. In the particular case of
    /// converting a character literal to u8, it is easy to avoid by just using a
    /// byte literal instead. As an added bonus, `b'a'` is even slightly shorter
    /// than `'a' as u8`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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

declare_lint_pass!(CharLitAsU8 => [CHAR_LIT_AS_U8]);

impl<'tcx> LateLintPass<'tcx> for CharLitAsU8 {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if !expr.span.from_expansion();
            if let ExprKind::Cast(e, _) = &expr.kind;
            if let ExprKind::Lit(l) = &e.kind;
            if let LitKind::Char(c) = l.node;
            if ty::Uint(UintTy::U8) == *cx.typeck_results().expr_ty(expr).kind();
            then {
                let mut applicability = Applicability::MachineApplicable;
                let snippet = snippet_with_applicability(cx, e.span, "'x'", &mut applicability);

                span_lint_and_then(
                    cx,
                    CHAR_LIT_AS_U8,
                    expr.span,
                    "casting a character literal to `u8` truncates",
                    |diag| {
                        diag.note("`char` is four bytes wide, but `u8` is a single byte");

                        if c.is_ascii() {
                            diag.span_suggestion(
                                expr.span,
                                "use a byte literal instead",
                                format!("b{}", snippet),
                                applicability,
                            );
                        }
                });
            }
        }
    }
}

declare_clippy_lint! {
    /// **What it does:**
    /// Checks for `as` casts between raw pointers without changing its mutability,
    /// namely `*const T` to `*const U` and `*mut T` to `*mut U`.
    ///
    /// **Why is this bad?**
    /// Though `as` casts between raw pointers is not terrible, `pointer::cast` is safer because
    /// it cannot accidentally change the pointer's mutability nor cast the pointer to other types like `usize`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
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

pub struct PtrAsPtr {
    msrv: Option<RustcVersion>,
}

impl PtrAsPtr {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(PtrAsPtr => [PTR_AS_PTR]);

impl<'tcx> LateLintPass<'tcx> for PtrAsPtr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !meets_msrv(self.msrv.as_ref(), &PTR_AS_PTR_MSRV) {
            return;
        }

        if expr.span.from_expansion() {
            return;
        }

        if_chain! {
            if let ExprKind::Cast(cast_expr, cast_to_hir_ty) = expr.kind;
            let (cast_from, cast_to) = (cx.typeck_results().expr_ty(cast_expr), cx.typeck_results().expr_ty(expr));
            if let ty::RawPtr(TypeAndMut { mutbl: from_mutbl, .. }) = cast_from.kind();
            if let ty::RawPtr(TypeAndMut { ty: to_pointee_ty, mutbl: to_mutbl }) = cast_to.kind();
            if matches!((from_mutbl, to_mutbl),
                (Mutability::Not, Mutability::Not) | (Mutability::Mut, Mutability::Mut));
            // The `U` in `pointer::cast` have to be `Sized`
            // as explained here: https://github.com/rust-lang/rust/issues/60602.
            if to_pointee_ty.is_sized(cx.tcx.at(expr.span), cx.param_env);
            then {
                let mut applicability = Applicability::MachineApplicable;
                let cast_expr_sugg = Sugg::hir_with_applicability(cx, cast_expr, "_", &mut applicability);
                let turbofish = match &cast_to_hir_ty.kind {
                        TyKind::Infer => Cow::Borrowed(""),
                        TyKind::Ptr(mut_ty) if matches!(mut_ty.ty.kind, TyKind::Infer) => Cow::Borrowed(""),
                        _ => Cow::Owned(format!("::<{}>", to_pointee_ty)),
                    };
                span_lint_and_sugg(
                    cx,
                    PTR_AS_PTR,
                    expr.span,
                    "`as` casting between raw pointers without changing its mutability",
                    "try `pointer::cast`, a safer alternative",
                    format!("{}.cast{}()", cast_expr_sugg.maybe_par(), turbofish),
                    applicability,
                );
            }
        }
    }

    extract_msrv_attr!(LateContext);
}
