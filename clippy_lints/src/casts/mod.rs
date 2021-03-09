use std::borrow::Cow;

use if_chain::if_chain;
use rustc_ast::{LitFloatType, LitIntType, LitKind};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, GenericArg, Lit, MutTy, Mutability, TyKind, UnOp};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, FloatTy, InferTy, IntTy, Ty, TyCtxt, TypeAndMut, UintTy};
use rustc_semver::RustcVersion;
use rustc_session::{declare_lint_pass, declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::sym;
use rustc_target::abi::LayoutOf;

use crate::consts::{constant, Constant};
use crate::utils::sugg::Sugg;
use crate::utils::{
    in_constant, is_hir_ty_cfg_dependant, meets_msrv, method_chain_args, numeric_literal::NumericLiteral, sext,
    snippet_opt, snippet_with_applicability, span_lint, span_lint_and_sugg, span_lint_and_then,
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

/// Returns the size in bits of an integral type.
/// Will return 0 if the type is not an int or uint variant
fn int_ty_to_nbits(typ: Ty<'_>, tcx: TyCtxt<'_>) -> u64 {
    match typ.kind() {
        ty::Int(i) => match i {
            IntTy::Isize => tcx.data_layout.pointer_size.bits(),
            IntTy::I8 => 8,
            IntTy::I16 => 16,
            IntTy::I32 => 32,
            IntTy::I64 => 64,
            IntTy::I128 => 128,
        },
        ty::Uint(i) => match i {
            UintTy::Usize => tcx.data_layout.pointer_size.bits(),
            UintTy::U8 => 8,
            UintTy::U16 => 16,
            UintTy::U32 => 32,
            UintTy::U64 => 64,
            UintTy::U128 => 128,
        },
        _ => 0,
    }
}

fn span_precision_loss_lint(cx: &LateContext<'_>, expr: &Expr<'_>, cast_from: Ty<'_>, cast_to_f64: bool) {
    let mantissa_nbits = if cast_to_f64 { 52 } else { 23 };
    let arch_dependent = is_isize_or_usize(cast_from) && cast_to_f64;
    let arch_dependent_str = "on targets with 64-bit wide pointers ";
    let from_nbits_str = if arch_dependent {
        "64".to_owned()
    } else if is_isize_or_usize(cast_from) {
        "32 or 64".to_owned()
    } else {
        int_ty_to_nbits(cast_from, cx.tcx).to_string()
    };
    span_lint(
        cx,
        CAST_PRECISION_LOSS,
        expr.span,
        &format!(
            "casting `{0}` to `{1}` causes a loss of precision {2}(`{0}` is {3} bits wide, \
             but `{1}`'s mantissa is only {4} bits wide)",
            cast_from,
            if cast_to_f64 { "f64" } else { "f32" },
            if arch_dependent { arch_dependent_str } else { "" },
            from_nbits_str,
            mantissa_nbits
        ),
    );
}

fn should_strip_parens(op: &Expr<'_>, snip: &str) -> bool {
    if let ExprKind::Binary(_, _, _) = op.kind {
        if snip.starts_with('(') && snip.ends_with(')') {
            return true;
        }
    }
    false
}

fn span_lossless_lint(cx: &LateContext<'_>, expr: &Expr<'_>, op: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    // Do not suggest using From in consts/statics until it is valid to do so (see #2267).
    if in_constant(cx, expr.hir_id) {
        return;
    }
    // The suggestion is to use a function call, so if the original expression
    // has parens on the outside, they are no longer needed.
    let mut applicability = Applicability::MachineApplicable;
    let opt = snippet_opt(cx, op.span);
    let sugg = opt.as_ref().map_or_else(
        || {
            applicability = Applicability::HasPlaceholders;
            ".."
        },
        |snip| {
            if should_strip_parens(op, snip) {
                &snip[1..snip.len() - 1]
            } else {
                snip.as_str()
            }
        },
    );

    span_lint_and_sugg(
        cx,
        CAST_LOSSLESS,
        expr.span,
        &format!(
            "casting `{}` to `{}` may become silently lossy if you later change the type",
            cast_from, cast_to
        ),
        "try",
        format!("{}::from({})", cast_to, sugg),
        applicability,
    );
}

enum ArchSuffix {
    _32,
    _64,
    None,
}

fn check_loss_of_sign(cx: &LateContext<'_>, expr: &Expr<'_>, op: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    if !cast_from.is_signed() || cast_to.is_signed() {
        return;
    }

    // don't lint for positive constants
    let const_val = constant(cx, &cx.typeck_results(), op);
    if_chain! {
        if let Some((Constant::Int(n), _)) = const_val;
        if let ty::Int(ity) = *cast_from.kind();
        if sext(cx.tcx, n, ity) >= 0;
        then {
            return
        }
    }

    // don't lint for the result of methods that always return non-negative values
    if let ExprKind::MethodCall(ref path, _, _, _) = op.kind {
        let mut method_name = path.ident.name.as_str();
        let allowed_methods = ["abs", "checked_abs", "rem_euclid", "checked_rem_euclid"];

        if_chain! {
            if method_name == "unwrap";
            if let Some(arglist) = method_chain_args(op, &["unwrap"]);
            if let ExprKind::MethodCall(ref inner_path, _, _, _) = &arglist[0][0].kind;
            then {
                method_name = inner_path.ident.name.as_str();
            }
        }

        if allowed_methods.iter().any(|&name| method_name == name) {
            return;
        }
    }

    span_lint(
        cx,
        CAST_SIGN_LOSS,
        expr.span,
        &format!(
            "casting `{}` to `{}` may lose the sign of the value",
            cast_from, cast_to
        ),
    );
}

fn check_truncation_and_wrapping(cx: &LateContext<'_>, expr: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    let arch_64_suffix = " on targets with 64-bit wide pointers";
    let arch_32_suffix = " on targets with 32-bit wide pointers";
    let cast_unsigned_to_signed = !cast_from.is_signed() && cast_to.is_signed();
    let from_nbits = int_ty_to_nbits(cast_from, cx.tcx);
    let to_nbits = int_ty_to_nbits(cast_to, cx.tcx);
    let (span_truncation, suffix_truncation, span_wrap, suffix_wrap) =
        match (is_isize_or_usize(cast_from), is_isize_or_usize(cast_to)) {
            (true, true) | (false, false) => (
                to_nbits < from_nbits,
                ArchSuffix::None,
                to_nbits == from_nbits && cast_unsigned_to_signed,
                ArchSuffix::None,
            ),
            (true, false) => (
                to_nbits <= 32,
                if to_nbits == 32 {
                    ArchSuffix::_64
                } else {
                    ArchSuffix::None
                },
                to_nbits <= 32 && cast_unsigned_to_signed,
                ArchSuffix::_32,
            ),
            (false, true) => (
                from_nbits == 64,
                ArchSuffix::_32,
                cast_unsigned_to_signed,
                if from_nbits == 64 {
                    ArchSuffix::_64
                } else {
                    ArchSuffix::_32
                },
            ),
        };
    if span_truncation {
        span_lint(
            cx,
            CAST_POSSIBLE_TRUNCATION,
            expr.span,
            &format!(
                "casting `{}` to `{}` may truncate the value{}",
                cast_from,
                cast_to,
                match suffix_truncation {
                    ArchSuffix::_32 => arch_32_suffix,
                    ArchSuffix::_64 => arch_64_suffix,
                    ArchSuffix::None => "",
                }
            ),
        );
    }
    if span_wrap {
        span_lint(
            cx,
            CAST_POSSIBLE_WRAP,
            expr.span,
            &format!(
                "casting `{}` to `{}` may wrap around the value{}",
                cast_from,
                cast_to,
                match suffix_wrap {
                    ArchSuffix::_32 => arch_32_suffix,
                    ArchSuffix::_64 => arch_64_suffix,
                    ArchSuffix::None => "",
                }
            ),
        );
    }
}

fn check_lossless(cx: &LateContext<'_>, expr: &Expr<'_>, op: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    let cast_signed_to_unsigned = cast_from.is_signed() && !cast_to.is_signed();
    let from_nbits = int_ty_to_nbits(cast_from, cx.tcx);
    let to_nbits = int_ty_to_nbits(cast_to, cx.tcx);
    if !is_isize_or_usize(cast_from) && !is_isize_or_usize(cast_to) && from_nbits < to_nbits && !cast_signed_to_unsigned
    {
        span_lossless_lint(cx, expr, op, cast_from, cast_to);
    }
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

/// Check if the given type is either `core::ffi::c_void` or
/// one of the platform specific `libc::<platform>::c_void` of libc.
fn is_c_void(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    if let ty::Adt(adt, _) = ty.kind() {
        let names = cx.get_def_path(adt.did);

        if names.is_empty() {
            return false;
        }
        if names[0] == sym::libc || names[0] == sym::core && *names.last().unwrap() == sym!(c_void) {
            return true;
        }
    }
    false
}

/// Returns the mantissa bits wide of a fp type.
/// Will return 0 if the type is not a fp
fn fp_ty_mantissa_nbits(typ: Ty<'_>) -> u32 {
    match typ.kind() {
        ty::Float(FloatTy::F32) => 23,
        ty::Float(FloatTy::F64) | ty::Infer(InferTy::FloatVar(_)) => 52,
        _ => 0,
    }
}

impl<'tcx> LateLintPass<'tcx> for Casts {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }
        if let ExprKind::Cast(ref ex, cast_to) = expr.kind {
            if is_hir_ty_cfg_dependant(cx, cast_to) {
                return;
            }
            let (cast_from, cast_to) = (cx.typeck_results().expr_ty(ex), cx.typeck_results().expr_ty(expr));
            lint_fn_to_numeric_cast(cx, expr, ex, cast_from, cast_to);
            if let Some(lit) = get_numeric_literal(ex) {
                let literal_str = snippet_opt(cx, ex.span).unwrap_or_default();

                if_chain! {
                    if let LitKind::Int(n, _) = lit.node;
                    if let Some(src) = snippet_opt(cx, lit.span);
                    if cast_to.is_floating_point();
                    if let Some(num_lit) = NumericLiteral::from_lit_kind(&src, &lit.node);
                    let from_nbits = 128 - n.leading_zeros();
                    let to_nbits = fp_ty_mantissa_nbits(cast_to);
                    if from_nbits != 0 && to_nbits != 0 && from_nbits <= to_nbits && num_lit.is_decimal();
                    then {
                        let literal_str = if is_unary_neg(ex) { format!("-{}", num_lit.integer) } else { num_lit.integer.into() };
                        show_unnecessary_cast(cx, expr, &literal_str, cast_from, cast_to);
                        return;
                    }
                }

                match lit.node {
                    LitKind::Int(_, LitIntType::Unsuffixed) if cast_to.is_integral() => {
                        show_unnecessary_cast(cx, expr, &literal_str, cast_from, cast_to);
                    },
                    LitKind::Float(_, LitFloatType::Unsuffixed) if cast_to.is_floating_point() => {
                        show_unnecessary_cast(cx, expr, &literal_str, cast_from, cast_to);
                    },
                    LitKind::Int(_, LitIntType::Unsuffixed) | LitKind::Float(_, LitFloatType::Unsuffixed) => {},
                    _ => {
                        if cast_from.kind() == cast_to.kind() && !in_external_macro(cx.sess(), expr.span) {
                            span_lint(
                                cx,
                                UNNECESSARY_CAST,
                                expr.span,
                                &format!(
                                    "casting to the same type is unnecessary (`{}` -> `{}`)",
                                    cast_from, cast_to
                                ),
                            );
                        }
                    },
                }
            }
            if cast_from.is_numeric() && cast_to.is_numeric() && !in_external_macro(cx.sess(), expr.span) {
                lint_numeric_casts(cx, expr, ex, cast_from, cast_to);
            }

            lint_cast_ptr_alignment(cx, expr, cast_from, cast_to);
        } else if let ExprKind::MethodCall(method_path, _, args, _) = expr.kind {
            if_chain! {
            if method_path.ident.name == sym!(cast);
            if let Some(generic_args) = method_path.args;
            if let [GenericArg::Type(cast_to)] = generic_args.args;
            // There probably is no obvious reason to do this, just to be consistent with `as` cases.
            if !is_hir_ty_cfg_dependant(cx, cast_to);
            then {
                let (cast_from, cast_to) =
                    (cx.typeck_results().expr_ty(&args[0]), cx.typeck_results().expr_ty(expr));
                lint_cast_ptr_alignment(cx, expr, cast_from, cast_to);
            }
            }
        }
    }
}

fn is_unary_neg(expr: &Expr<'_>) -> bool {
    matches!(expr.kind, ExprKind::Unary(UnOp::Neg, _))
}

fn get_numeric_literal<'e>(expr: &'e Expr<'e>) -> Option<&'e Lit> {
    match expr.kind {
        ExprKind::Lit(ref lit) => Some(lit),
        ExprKind::Unary(UnOp::Neg, e) => {
            if let ExprKind::Lit(ref lit) = e.kind {
                Some(lit)
            } else {
                None
            }
        },
        _ => None,
    }
}

fn show_unnecessary_cast(cx: &LateContext<'_>, expr: &Expr<'_>, literal_str: &str, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    let literal_kind_name = if cast_from.is_integral() { "integer" } else { "float" };
    span_lint_and_sugg(
        cx,
        UNNECESSARY_CAST,
        expr.span,
        &format!("casting {} literal to `{}` is unnecessary", literal_kind_name, cast_to),
        "try",
        format!("{}_{}", literal_str.trim_end_matches('.'), cast_to),
        Applicability::MachineApplicable,
    );
}

fn lint_numeric_casts<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    cast_expr: &Expr<'_>,
    cast_from: Ty<'tcx>,
    cast_to: Ty<'tcx>,
) {
    match (cast_from.is_integral(), cast_to.is_integral()) {
        (true, false) => {
            let from_nbits = int_ty_to_nbits(cast_from, cx.tcx);
            let to_nbits = if let ty::Float(FloatTy::F32) = cast_to.kind() {
                32
            } else {
                64
            };
            if is_isize_or_usize(cast_from) || from_nbits >= to_nbits {
                span_precision_loss_lint(cx, expr, cast_from, to_nbits == 64);
            }
            if from_nbits < to_nbits {
                span_lossless_lint(cx, expr, cast_expr, cast_from, cast_to);
            }
        },
        (false, true) => {
            span_lint(
                cx,
                CAST_POSSIBLE_TRUNCATION,
                expr.span,
                &format!("casting `{}` to `{}` may truncate the value", cast_from, cast_to),
            );
            if !cast_to.is_signed() {
                span_lint(
                    cx,
                    CAST_SIGN_LOSS,
                    expr.span,
                    &format!(
                        "casting `{}` to `{}` may lose the sign of the value",
                        cast_from, cast_to
                    ),
                );
            }
        },
        (true, true) => {
            check_loss_of_sign(cx, expr, cast_expr, cast_from, cast_to);
            check_truncation_and_wrapping(cx, expr, cast_from, cast_to);
            check_lossless(cx, expr, cast_expr, cast_from, cast_to);
        },
        (false, false) => {
            if let (&ty::Float(FloatTy::F64), &ty::Float(FloatTy::F32)) = (&cast_from.kind(), &cast_to.kind()) {
                span_lint(
                    cx,
                    CAST_POSSIBLE_TRUNCATION,
                    expr.span,
                    "casting `f64` to `f32` may truncate the value",
                );
            }
            if let (&ty::Float(FloatTy::F32), &ty::Float(FloatTy::F64)) = (&cast_from.kind(), &cast_to.kind()) {
                span_lossless_lint(cx, expr, cast_expr, cast_from, cast_to);
            }
        },
    }
}

fn lint_cast_ptr_alignment<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, cast_from: Ty<'tcx>, cast_to: Ty<'tcx>) {
    if_chain! {
        if let ty::RawPtr(from_ptr_ty) = &cast_from.kind();
        if let ty::RawPtr(to_ptr_ty) = &cast_to.kind();
        if let Ok(from_layout) = cx.layout_of(from_ptr_ty.ty);
        if let Ok(to_layout) = cx.layout_of(to_ptr_ty.ty);
        if from_layout.align.abi < to_layout.align.abi;
        // with c_void, we inherently need to trust the user
        if !is_c_void(cx, from_ptr_ty.ty);
        // when casting from a ZST, we don't know enough to properly lint
        if !from_layout.is_zst();
        then {
            span_lint(
                cx,
                CAST_PTR_ALIGNMENT,
                expr.span,
                &format!(
                    "casting from `{}` to a more-strictly-aligned pointer (`{}`) ({} < {} bytes)",
                    cast_from,
                    cast_to,
                    from_layout.align.abi.bytes(),
                    to_layout.align.abi.bytes(),
                ),
            );
        }
    }
}

fn lint_fn_to_numeric_cast(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    cast_expr: &Expr<'_>,
    cast_from: Ty<'_>,
    cast_to: Ty<'_>,
) {
    // We only want to check casts to `ty::Uint` or `ty::Int`
    match cast_to.kind() {
        ty::Uint(_) | ty::Int(..) => { /* continue on */ },
        _ => return,
    }
    match cast_from.kind() {
        ty::FnDef(..) | ty::FnPtr(_) => {
            let mut applicability = Applicability::MaybeIncorrect;
            let from_snippet = snippet_with_applicability(cx, cast_expr.span, "x", &mut applicability);

            let to_nbits = int_ty_to_nbits(cast_to, cx.tcx);
            if to_nbits < cx.tcx.data_layout.pointer_size.bits() {
                span_lint_and_sugg(
                    cx,
                    FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
                    expr.span,
                    &format!(
                        "casting function pointer `{}` to `{}`, which truncates the value",
                        from_snippet, cast_to
                    ),
                    "try",
                    format!("{} as usize", from_snippet),
                    applicability,
                );
            } else if *cast_to.kind() != ty::Uint(UintTy::Usize) {
                span_lint_and_sugg(
                    cx,
                    FN_TO_NUMERIC_CAST,
                    expr.span,
                    &format!("casting function pointer `{}` to `{}`", from_snippet, cast_to),
                    "try",
                    format!("{} as usize", from_snippet),
                    applicability,
                );
            }
        },
        _ => {},
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

fn is_isize_or_usize(typ: Ty<'_>) -> bool {
    matches!(typ.kind(), ty::Int(IntTy::Isize) | ty::Uint(UintTy::Usize))
}
