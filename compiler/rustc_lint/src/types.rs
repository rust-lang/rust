use crate::{
    fluent_generated as fluent,
    lints::{
        AtomicOrderingFence, AtomicOrderingLoad, AtomicOrderingStore, ImproperCTypes,
        InvalidAtomicOrderingDiag, InvalidNanComparisons, InvalidNanComparisonsSuggestion,
        OnlyCastu8ToChar, OverflowingBinHex, OverflowingBinHexSign, OverflowingBinHexSub,
        OverflowingInt, OverflowingIntHelp, OverflowingLiteral, OverflowingUInt,
        RangeEndpointOutOfRange, UnusedComparisons, UseInclusiveRange, VariantSizeDifferencesDiag,
    },
};
use crate::{LateContext, LateLintPass, LintContext};
use rustc_ast as ast;
use rustc_attr as attr;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::DiagnosticMessage;
use rustc_hir as hir;
use rustc_hir::{is_range_literal, Expr, ExprKind, Node};
use rustc_middle::ty::layout::{IntegerExt, LayoutOf, SizeSkeleton};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{
    self, AdtKind, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
};
use rustc_span::def_id::LocalDefId;
use rustc_span::source_map;
use rustc_span::symbol::sym;
use rustc_span::{Span, Symbol};
use rustc_target::abi::{Abi, Size, WrappingRange};
use rustc_target::abi::{Integer, TagEncoding, Variants};
use rustc_target::spec::abi::Abi as SpecAbi;

use std::iter;
use std::ops::ControlFlow;

declare_lint! {
    /// The `unused_comparisons` lint detects comparisons made useless by
    /// limits of the types involved.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn foo(x: u8) {
    ///     x >= 0;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// A useless comparison may indicate a mistake, and should be fixed or
    /// removed.
    UNUSED_COMPARISONS,
    Warn,
    "comparisons made useless by limits of the types involved"
}

declare_lint! {
    /// The `overflowing_literals` lint detects literal out of range for its
    /// type.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// let x: u8 = 1000;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is usually a mistake to use a literal that overflows the type where
    /// it is used. Either use a literal that is within range, or change the
    /// type to be within the range of the literal.
    OVERFLOWING_LITERALS,
    Deny,
    "literal out of range for its type"
}

declare_lint! {
    /// The `variant_size_differences` lint detects enums with widely varying
    /// variant sizes.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(variant_size_differences)]
    /// enum En {
    ///     V0(u8),
    ///     VBig([u8; 1024]),
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It can be a mistake to add a variant to an enum that is much larger
    /// than the other variants, bloating the overall size required for all
    /// variants. This can impact performance and memory usage. This is
    /// triggered if one variant is more than 3 times larger than the
    /// second-largest variant.
    ///
    /// Consider placing the large variant's contents on the heap (for example
    /// via [`Box`]) to keep the overall size of the enum itself down.
    ///
    /// This lint is "allow" by default because it can be noisy, and may not be
    /// an actual problem. Decisions about this should be guided with
    /// profiling and benchmarking.
    ///
    /// [`Box`]: https://doc.rust-lang.org/std/boxed/index.html
    VARIANT_SIZE_DIFFERENCES,
    Allow,
    "detects enums with widely varying variant sizes"
}

declare_lint! {
    /// The `invalid_nan_comparisons` lint checks comparison with `f32::NAN` or `f64::NAN`
    /// as one of the operand.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let a = 2.3f32;
    /// if a == f32::NAN {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// NaN does not compare meaningfully to anything – not
    /// even itself – so those comparisons are always false.
    INVALID_NAN_COMPARISONS,
    Warn,
    "detects invalid floating point NaN comparisons"
}

#[derive(Copy, Clone)]
pub struct TypeLimits {
    /// Id of the last visited negated expression
    negated_expr_id: Option<hir::HirId>,
}

impl_lint_pass!(TypeLimits => [UNUSED_COMPARISONS, OVERFLOWING_LITERALS, INVALID_NAN_COMPARISONS]);

impl TypeLimits {
    pub fn new() -> TypeLimits {
        TypeLimits { negated_expr_id: None }
    }
}

/// Attempts to special-case the overflowing literal lint when it occurs as a range endpoint (`expr..MAX+1`).
/// Returns `true` iff the lint was emitted.
fn lint_overflowing_range_endpoint<'tcx>(
    cx: &LateContext<'tcx>,
    lit: &hir::Lit,
    lit_val: u128,
    max: u128,
    expr: &'tcx hir::Expr<'tcx>,
    ty: &str,
) -> bool {
    // Look past casts to support cases like `0..256 as u8`
    let (expr, lit_span) = if let Node::Expr(par_expr) = cx.tcx.hir().get(cx.tcx.hir().parent_id(expr.hir_id))
      && let ExprKind::Cast(_, _) = par_expr.kind {
        (par_expr, expr.span)
    } else {
        (expr, expr.span)
    };

    // We only want to handle exclusive (`..`) ranges,
    // which are represented as `ExprKind::Struct`.
    let par_id = cx.tcx.hir().parent_id(expr.hir_id);
    let Node::ExprField(field) = cx.tcx.hir().get(par_id) else { return false };
    let Node::Expr(struct_expr) = cx.tcx.hir().get_parent(field.hir_id) else { return false };
    if !is_range_literal(struct_expr) {
        return false;
    };
    let ExprKind::Struct(_, eps, _) = &struct_expr.kind else { return false };
    if eps.len() != 2 {
        return false;
    }

    // We can suggest using an inclusive range
    // (`..=`) instead only if it is the `end` that is
    // overflowing and only by 1.
    if !(eps[1].expr.hir_id == expr.hir_id && lit_val - 1 == max) {
        return false;
    };

    use rustc_ast::{LitIntType, LitKind};
    let suffix = match lit.node {
        LitKind::Int(_, LitIntType::Signed(s)) => s.name_str(),
        LitKind::Int(_, LitIntType::Unsigned(s)) => s.name_str(),
        LitKind::Int(_, LitIntType::Unsuffixed) => "",
        _ => bug!(),
    };

    let sub_sugg = if expr.span.lo() == lit_span.lo() {
        let Ok(start) = cx.sess().source_map().span_to_snippet(eps[0].span) else { return false };
        UseInclusiveRange::WithoutParen {
            sugg: struct_expr.span.shrink_to_lo().to(lit_span.shrink_to_hi()),
            start,
            literal: lit_val - 1,
            suffix,
        }
    } else {
        UseInclusiveRange::WithParen {
            eq_sugg: expr.span.shrink_to_lo(),
            lit_sugg: lit_span,
            literal: lit_val - 1,
            suffix,
        }
    };

    cx.emit_spanned_lint(
        OVERFLOWING_LITERALS,
        struct_expr.span,
        RangeEndpointOutOfRange { ty, sub: sub_sugg },
    );

    // We've just emitted a lint, special cased for `(...)..MAX+1` ranges,
    // return `true` so the callers don't also emit a lint
    true
}

// For `isize` & `usize`, be conservative with the warnings, so that the
// warnings are consistent between 32- and 64-bit platforms.
fn int_ty_range(int_ty: ty::IntTy) -> (i128, i128) {
    match int_ty {
        ty::IntTy::Isize => (i64::MIN.into(), i64::MAX.into()),
        ty::IntTy::I8 => (i8::MIN.into(), i8::MAX.into()),
        ty::IntTy::I16 => (i16::MIN.into(), i16::MAX.into()),
        ty::IntTy::I32 => (i32::MIN.into(), i32::MAX.into()),
        ty::IntTy::I64 => (i64::MIN.into(), i64::MAX.into()),
        ty::IntTy::I128 => (i128::MIN, i128::MAX),
    }
}

fn uint_ty_range(uint_ty: ty::UintTy) -> (u128, u128) {
    let max = match uint_ty {
        ty::UintTy::Usize => u64::MAX.into(),
        ty::UintTy::U8 => u8::MAX.into(),
        ty::UintTy::U16 => u16::MAX.into(),
        ty::UintTy::U32 => u32::MAX.into(),
        ty::UintTy::U64 => u64::MAX.into(),
        ty::UintTy::U128 => u128::MAX,
    };
    (0, max)
}

fn get_bin_hex_repr(cx: &LateContext<'_>, lit: &hir::Lit) -> Option<String> {
    let src = cx.sess().source_map().span_to_snippet(lit.span).ok()?;
    let firstch = src.chars().next()?;

    if firstch == '0' {
        match src.chars().nth(1) {
            Some('x' | 'b') => return Some(src),
            _ => return None,
        }
    }

    None
}

fn report_bin_hex_error(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    ty: attr::IntType,
    size: Size,
    repr_str: String,
    val: u128,
    negative: bool,
) {
    let (t, actually) = match ty {
        attr::IntType::SignedInt(t) => {
            let actually = if negative {
                -(size.sign_extend(val) as i128)
            } else {
                size.sign_extend(val) as i128
            };
            (t.name_str(), actually.to_string())
        }
        attr::IntType::UnsignedInt(t) => {
            let actually = size.truncate(val);
            (t.name_str(), actually.to_string())
        }
    };
    let sign =
        if negative { OverflowingBinHexSign::Negative } else { OverflowingBinHexSign::Positive };
    let sub = get_type_suggestion(cx.typeck_results().node_type(expr.hir_id), val, negative).map(
        |suggestion_ty| {
            if let Some(pos) = repr_str.chars().position(|c| c == 'i' || c == 'u') {
                let (sans_suffix, _) = repr_str.split_at(pos);
                OverflowingBinHexSub::Suggestion { span: expr.span, suggestion_ty, sans_suffix }
            } else {
                OverflowingBinHexSub::Help { suggestion_ty }
            }
        },
    );
    cx.emit_spanned_lint(
        OVERFLOWING_LITERALS,
        expr.span,
        OverflowingBinHex { ty: t, lit: repr_str.clone(), dec: val, actually, sign, sub },
    )
}

// This function finds the next fitting type and generates a suggestion string.
// It searches for fitting types in the following way (`X < Y`):
//  - `iX`: if literal fits in `uX` => `uX`, else => `iY`
//  - `-iX` => `iY`
//  - `uX` => `uY`
//
// No suggestion for: `isize`, `usize`.
fn get_type_suggestion(t: Ty<'_>, val: u128, negative: bool) -> Option<&'static str> {
    use ty::IntTy::*;
    use ty::UintTy::*;
    macro_rules! find_fit {
        ($ty:expr, $val:expr, $negative:expr,
         $($type:ident => [$($utypes:expr),*] => [$($itypes:expr),*]),+) => {
            {
                let _neg = if negative { 1 } else { 0 };
                match $ty {
                    $($type => {
                        $(if !negative && val <= uint_ty_range($utypes).1 {
                            return Some($utypes.name_str())
                        })*
                        $(if val <= int_ty_range($itypes).1 as u128 + _neg {
                            return Some($itypes.name_str())
                        })*
                        None
                    },)+
                    _ => None
                }
            }
        }
    }
    match t.kind() {
        ty::Int(i) => find_fit!(i, val, negative,
                      I8 => [U8] => [I16, I32, I64, I128],
                      I16 => [U16] => [I32, I64, I128],
                      I32 => [U32] => [I64, I128],
                      I64 => [U64] => [I128],
                      I128 => [U128] => []),
        ty::Uint(u) => find_fit!(u, val, negative,
                      U8 => [U8, U16, U32, U64, U128] => [],
                      U16 => [U16, U32, U64, U128] => [],
                      U32 => [U32, U64, U128] => [],
                      U64 => [U64, U128] => [],
                      U128 => [U128] => []),
        _ => None,
    }
}

fn lint_int_literal<'tcx>(
    cx: &LateContext<'tcx>,
    type_limits: &TypeLimits,
    e: &'tcx hir::Expr<'tcx>,
    lit: &hir::Lit,
    t: ty::IntTy,
    v: u128,
) {
    let int_type = t.normalize(cx.sess().target.pointer_width);
    let (min, max) = int_ty_range(int_type);
    let max = max as u128;
    let negative = type_limits.negated_expr_id == Some(e.hir_id);

    // Detect literal value out of range [min, max] inclusive
    // avoiding use of -min to prevent overflow/panic
    if (negative && v > max + 1) || (!negative && v > max) {
        if let Some(repr_str) = get_bin_hex_repr(cx, lit) {
            report_bin_hex_error(
                cx,
                e,
                attr::IntType::SignedInt(ty::ast_int_ty(t)),
                Integer::from_int_ty(cx, t).size(),
                repr_str,
                v,
                negative,
            );
            return;
        }

        if lint_overflowing_range_endpoint(cx, lit, v, max, e, t.name_str()) {
            // The overflowing literal lint was emitted by `lint_overflowing_range_endpoint`.
            return;
        }

        let lit = cx
            .sess()
            .source_map()
            .span_to_snippet(lit.span)
            .expect("must get snippet from literal");
        let help = get_type_suggestion(cx.typeck_results().node_type(e.hir_id), v, negative)
            .map(|suggestion_ty| OverflowingIntHelp { suggestion_ty });

        cx.emit_spanned_lint(
            OVERFLOWING_LITERALS,
            e.span,
            OverflowingInt { ty: t.name_str(), lit, min, max, help },
        );
    }
}

fn lint_uint_literal<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx hir::Expr<'tcx>,
    lit: &hir::Lit,
    t: ty::UintTy,
) {
    let uint_type = t.normalize(cx.sess().target.pointer_width);
    let (min, max) = uint_ty_range(uint_type);
    let lit_val: u128 = match lit.node {
        // _v is u8, within range by definition
        ast::LitKind::Byte(_v) => return,
        ast::LitKind::Int(v, _) => v,
        _ => bug!(),
    };
    if lit_val < min || lit_val > max {
        let parent_id = cx.tcx.hir().parent_id(e.hir_id);
        if let Node::Expr(par_e) = cx.tcx.hir().get(parent_id) {
            match par_e.kind {
                hir::ExprKind::Cast(..) => {
                    if let ty::Char = cx.typeck_results().expr_ty(par_e).kind() {
                        cx.emit_spanned_lint(
                            OVERFLOWING_LITERALS,
                            par_e.span,
                            OnlyCastu8ToChar { span: par_e.span, literal: lit_val },
                        );
                        return;
                    }
                }
                _ => {}
            }
        }
        if lint_overflowing_range_endpoint(cx, lit, lit_val, max, e, t.name_str()) {
            // The overflowing literal lint was emitted by `lint_overflowing_range_endpoint`.
            return;
        }
        if let Some(repr_str) = get_bin_hex_repr(cx, lit) {
            report_bin_hex_error(
                cx,
                e,
                attr::IntType::UnsignedInt(ty::ast_uint_ty(t)),
                Integer::from_uint_ty(cx, t).size(),
                repr_str,
                lit_val,
                false,
            );
            return;
        }
        cx.emit_spanned_lint(
            OVERFLOWING_LITERALS,
            e.span,
            OverflowingUInt {
                ty: t.name_str(),
                lit: cx
                    .sess()
                    .source_map()
                    .span_to_snippet(lit.span)
                    .expect("must get snippet from literal"),
                min,
                max,
            },
        );
    }
}

fn lint_literal<'tcx>(
    cx: &LateContext<'tcx>,
    type_limits: &TypeLimits,
    e: &'tcx hir::Expr<'tcx>,
    lit: &hir::Lit,
) {
    match *cx.typeck_results().node_type(e.hir_id).kind() {
        ty::Int(t) => {
            match lit.node {
                ast::LitKind::Int(v, ast::LitIntType::Signed(_) | ast::LitIntType::Unsuffixed) => {
                    lint_int_literal(cx, type_limits, e, lit, t, v)
                }
                _ => bug!(),
            };
        }
        ty::Uint(t) => lint_uint_literal(cx, e, lit, t),
        ty::Float(t) => {
            let is_infinite = match lit.node {
                ast::LitKind::Float(v, _) => match t {
                    ty::FloatTy::F32 => v.as_str().parse().map(f32::is_infinite),
                    ty::FloatTy::F64 => v.as_str().parse().map(f64::is_infinite),
                },
                _ => bug!(),
            };
            if is_infinite == Ok(true) {
                cx.emit_spanned_lint(
                    OVERFLOWING_LITERALS,
                    e.span,
                    OverflowingLiteral {
                        ty: t.name_str(),
                        lit: cx
                            .sess()
                            .source_map()
                            .span_to_snippet(lit.span)
                            .expect("must get snippet from literal"),
                    },
                );
            }
        }
        _ => {}
    }
}

fn lint_nan<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx hir::Expr<'tcx>,
    binop: hir::BinOp,
    l: &'tcx hir::Expr<'tcx>,
    r: &'tcx hir::Expr<'tcx>,
) {
    fn is_nan(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
        let expr = expr.peel_blocks().peel_borrows();
        match expr.kind {
            ExprKind::Path(qpath) => {
                let Some(def_id) = cx.typeck_results().qpath_res(&qpath, expr.hir_id).opt_def_id() else { return false; };

                matches!(cx.tcx.get_diagnostic_name(def_id), Some(sym::f32_nan | sym::f64_nan))
            }
            _ => false,
        }
    }

    fn eq_ne(
        e: &hir::Expr<'_>,
        l: &hir::Expr<'_>,
        r: &hir::Expr<'_>,
        f: impl FnOnce(Span, Span) -> InvalidNanComparisonsSuggestion,
    ) -> InvalidNanComparisons {
        let suggestion =
            if let Some(l_span) = l.span.find_ancestor_inside(e.span) &&
                let Some(r_span) = r.span.find_ancestor_inside(e.span) {
                f(l_span, r_span)
            } else {
                InvalidNanComparisonsSuggestion::Spanless
            };

        InvalidNanComparisons::EqNe { suggestion }
    }

    let lint = match binop.node {
        hir::BinOpKind::Eq | hir::BinOpKind::Ne if is_nan(cx, l) => {
            eq_ne(e, l, r, |l_span, r_span| InvalidNanComparisonsSuggestion::Spanful {
                nan_plus_binop: l_span.until(r_span),
                float: r_span.shrink_to_hi(),
                neg: (binop.node == hir::BinOpKind::Ne).then(|| r_span.shrink_to_lo()),
            })
        }
        hir::BinOpKind::Eq | hir::BinOpKind::Ne if is_nan(cx, r) => {
            eq_ne(e, l, r, |l_span, r_span| InvalidNanComparisonsSuggestion::Spanful {
                nan_plus_binop: l_span.shrink_to_hi().to(r_span),
                float: l_span.shrink_to_hi(),
                neg: (binop.node == hir::BinOpKind::Ne).then(|| l_span.shrink_to_lo()),
            })
        }
        hir::BinOpKind::Lt | hir::BinOpKind::Le | hir::BinOpKind::Gt | hir::BinOpKind::Ge
            if is_nan(cx, l) || is_nan(cx, r) =>
        {
            InvalidNanComparisons::LtLeGtGe
        }
        _ => return,
    };

    cx.emit_spanned_lint(INVALID_NAN_COMPARISONS, e.span, lint);
}

impl<'tcx> LateLintPass<'tcx> for TypeLimits {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx hir::Expr<'tcx>) {
        match e.kind {
            hir::ExprKind::Unary(hir::UnOp::Neg, ref expr) => {
                // propagate negation, if the negation itself isn't negated
                if self.negated_expr_id != Some(e.hir_id) {
                    self.negated_expr_id = Some(expr.hir_id);
                }
            }
            hir::ExprKind::Binary(binop, ref l, ref r) => {
                if is_comparison(binop) {
                    if !check_limits(cx, binop, &l, &r) {
                        cx.emit_spanned_lint(UNUSED_COMPARISONS, e.span, UnusedComparisons);
                    } else {
                        lint_nan(cx, e, binop, l, r);
                    }
                }
            }
            hir::ExprKind::Lit(ref lit) => lint_literal(cx, self, e, lit),
            _ => {}
        };

        fn is_valid<T: PartialOrd>(binop: hir::BinOp, v: T, min: T, max: T) -> bool {
            match binop.node {
                hir::BinOpKind::Lt => v > min && v <= max,
                hir::BinOpKind::Le => v >= min && v < max,
                hir::BinOpKind::Gt => v >= min && v < max,
                hir::BinOpKind::Ge => v > min && v <= max,
                hir::BinOpKind::Eq | hir::BinOpKind::Ne => v >= min && v <= max,
                _ => bug!(),
            }
        }

        fn rev_binop(binop: hir::BinOp) -> hir::BinOp {
            source_map::respan(
                binop.span,
                match binop.node {
                    hir::BinOpKind::Lt => hir::BinOpKind::Gt,
                    hir::BinOpKind::Le => hir::BinOpKind::Ge,
                    hir::BinOpKind::Gt => hir::BinOpKind::Lt,
                    hir::BinOpKind::Ge => hir::BinOpKind::Le,
                    _ => return binop,
                },
            )
        }

        fn check_limits(
            cx: &LateContext<'_>,
            binop: hir::BinOp,
            l: &hir::Expr<'_>,
            r: &hir::Expr<'_>,
        ) -> bool {
            let (lit, expr, swap) = match (&l.kind, &r.kind) {
                (&hir::ExprKind::Lit(_), _) => (l, r, true),
                (_, &hir::ExprKind::Lit(_)) => (r, l, false),
                _ => return true,
            };
            // Normalize the binop so that the literal is always on the RHS in
            // the comparison
            let norm_binop = if swap { rev_binop(binop) } else { binop };
            match *cx.typeck_results().node_type(expr.hir_id).kind() {
                ty::Int(int_ty) => {
                    let (min, max) = int_ty_range(int_ty);
                    let lit_val: i128 = match lit.kind {
                        hir::ExprKind::Lit(ref li) => match li.node {
                            ast::LitKind::Int(
                                v,
                                ast::LitIntType::Signed(_) | ast::LitIntType::Unsuffixed,
                            ) => v as i128,
                            _ => return true,
                        },
                        _ => bug!(),
                    };
                    is_valid(norm_binop, lit_val, min, max)
                }
                ty::Uint(uint_ty) => {
                    let (min, max): (u128, u128) = uint_ty_range(uint_ty);
                    let lit_val: u128 = match lit.kind {
                        hir::ExprKind::Lit(ref li) => match li.node {
                            ast::LitKind::Int(v, _) => v,
                            _ => return true,
                        },
                        _ => bug!(),
                    };
                    is_valid(norm_binop, lit_val, min, max)
                }
                _ => true,
            }
        }

        fn is_comparison(binop: hir::BinOp) -> bool {
            matches!(
                binop.node,
                hir::BinOpKind::Eq
                    | hir::BinOpKind::Lt
                    | hir::BinOpKind::Le
                    | hir::BinOpKind::Ne
                    | hir::BinOpKind::Ge
                    | hir::BinOpKind::Gt
            )
        }
    }
}

declare_lint! {
    /// The `improper_ctypes` lint detects incorrect use of types in foreign
    /// modules.
    ///
    /// ### Example
    ///
    /// ```rust
    /// extern "C" {
    ///     static STATIC: String;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The compiler has several checks to verify that types used in `extern`
    /// blocks are safe and follow certain rules to ensure proper
    /// compatibility with the foreign interfaces. This lint is issued when it
    /// detects a probable mistake in a definition. The lint usually should
    /// provide a description of the issue, along with possibly a hint on how
    /// to resolve it.
    IMPROPER_CTYPES,
    Warn,
    "proper use of libc types in foreign modules"
}

declare_lint_pass!(ImproperCTypesDeclarations => [IMPROPER_CTYPES]);

declare_lint! {
    /// The `improper_ctypes_definitions` lint detects incorrect use of
    /// [`extern` function] definitions.
    ///
    /// [`extern` function]: https://doc.rust-lang.org/reference/items/functions.html#extern-function-qualifier
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// pub extern "C" fn str_type(p: &str) { }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// There are many parameter and return types that may be specified in an
    /// `extern` function that are not compatible with the given ABI. This
    /// lint is an alert that these types should not be used. The lint usually
    /// should provide a description of the issue, along with possibly a hint
    /// on how to resolve it.
    IMPROPER_CTYPES_DEFINITIONS,
    Warn,
    "proper use of libc types in foreign item definitions"
}

declare_lint_pass!(ImproperCTypesDefinitions => [IMPROPER_CTYPES_DEFINITIONS]);

#[derive(Clone, Copy)]
pub(crate) enum CItemKind {
    Declaration,
    Definition,
}

struct ImproperCTypesVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    mode: CItemKind,
}

enum FfiResult<'tcx> {
    FfiSafe,
    FfiPhantom(Ty<'tcx>),
    FfiUnsafe { ty: Ty<'tcx>, reason: DiagnosticMessage, help: Option<DiagnosticMessage> },
}

pub(crate) fn nonnull_optimization_guaranteed<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::AdtDef<'tcx>,
) -> bool {
    tcx.has_attr(def.did(), sym::rustc_nonnull_optimization_guaranteed)
}

/// `repr(transparent)` structs can have a single non-ZST field, this function returns that
/// field.
pub fn transparent_newtype_field<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    variant: &'a ty::VariantDef,
) -> Option<&'a ty::FieldDef> {
    let param_env = tcx.param_env(variant.def_id);
    variant.fields.iter().find(|field| {
        let field_ty = tcx.type_of(field.did).subst_identity();
        let is_zst = tcx.layout_of(param_env.and(field_ty)).is_ok_and(|layout| layout.is_zst());
        !is_zst
    })
}

/// Is type known to be non-null?
fn ty_is_known_nonnull<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, mode: CItemKind) -> bool {
    let tcx = cx.tcx;
    match ty.kind() {
        ty::FnPtr(_) => true,
        ty::Ref(..) => true,
        ty::Adt(def, _) if def.is_box() && matches!(mode, CItemKind::Definition) => true,
        ty::Adt(def, substs) if def.repr().transparent() && !def.is_union() => {
            let marked_non_null = nonnull_optimization_guaranteed(tcx, *def);

            if marked_non_null {
                return true;
            }

            // `UnsafeCell` has its niche hidden.
            if def.is_unsafe_cell() {
                return false;
            }

            def.variants()
                .iter()
                .filter_map(|variant| transparent_newtype_field(cx.tcx, variant))
                .any(|field| ty_is_known_nonnull(cx, field.ty(tcx, substs), mode))
        }
        _ => false,
    }
}

/// Given a non-null scalar (or transparent) type `ty`, return the nullable version of that type.
/// If the type passed in was not scalar, returns None.
fn get_nullable_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    let tcx = cx.tcx;
    Some(match *ty.kind() {
        ty::Adt(field_def, field_substs) => {
            let inner_field_ty = {
                let mut first_non_zst_ty = field_def
                    .variants()
                    .iter()
                    .filter_map(|v| transparent_newtype_field(cx.tcx, v));
                debug_assert_eq!(
                    first_non_zst_ty.clone().count(),
                    1,
                    "Wrong number of fields for transparent type"
                );
                first_non_zst_ty
                    .next_back()
                    .expect("No non-zst fields in transparent type.")
                    .ty(tcx, field_substs)
            };
            return get_nullable_type(cx, inner_field_ty);
        }
        ty::Int(ty) => tcx.mk_mach_int(ty),
        ty::Uint(ty) => tcx.mk_mach_uint(ty),
        ty::RawPtr(ty_mut) => tcx.mk_ptr(ty_mut),
        // As these types are always non-null, the nullable equivalent of
        // Option<T> of these types are their raw pointer counterparts.
        ty::Ref(_region, ty, mutbl) => tcx.mk_ptr(ty::TypeAndMut { ty, mutbl }),
        ty::FnPtr(..) => {
            // There is no nullable equivalent for Rust's function pointers -- you
            // must use an Option<fn(..) -> _> to represent it.
            ty
        }

        // We should only ever reach this case if ty_is_known_nonnull is extended
        // to other types.
        ref unhandled => {
            debug!(
                "get_nullable_type: Unhandled scalar kind: {:?} while checking {:?}",
                unhandled, ty
            );
            return None;
        }
    })
}

/// Check if this enum can be safely exported based on the "nullable pointer optimization". If it
/// can, return the type that `ty` can be safely converted to, otherwise return `None`.
/// Currently restricted to function pointers, boxes, references, `core::num::NonZero*`,
/// `core::ptr::NonNull`, and `#[repr(transparent)]` newtypes.
/// FIXME: This duplicates code in codegen.
pub(crate) fn repr_nullable_ptr<'tcx>(
    cx: &LateContext<'tcx>,
    ty: Ty<'tcx>,
    ckind: CItemKind,
) -> Option<Ty<'tcx>> {
    debug!("is_repr_nullable_ptr(cx, ty = {:?})", ty);
    if let ty::Adt(ty_def, substs) = ty.kind() {
        let field_ty = match &ty_def.variants().raw[..] {
            [var_one, var_two] => match (&var_one.fields.raw[..], &var_two.fields.raw[..]) {
                ([], [field]) | ([field], []) => field.ty(cx.tcx, substs),
                _ => return None,
            },
            _ => return None,
        };

        if !ty_is_known_nonnull(cx, field_ty, ckind) {
            return None;
        }

        // At this point, the field's type is known to be nonnull and the parent enum is Option-like.
        // If the computed size for the field and the enum are different, the nonnull optimization isn't
        // being applied (and we've got a problem somewhere).
        let compute_size_skeleton = |t| SizeSkeleton::compute(t, cx.tcx, cx.param_env).unwrap();
        if !compute_size_skeleton(ty).same_size(compute_size_skeleton(field_ty)) {
            bug!("improper_ctypes: Option nonnull optimization not applied?");
        }

        // Return the nullable type this Option-like enum can be safely represented with.
        let field_ty_abi = &cx.layout_of(field_ty).unwrap().abi;
        if let Abi::Scalar(field_ty_scalar) = field_ty_abi {
            match field_ty_scalar.valid_range(cx) {
                WrappingRange { start: 0, end }
                    if end == field_ty_scalar.size(&cx.tcx).unsigned_int_max() - 1 =>
                {
                    return Some(get_nullable_type(cx, field_ty).unwrap());
                }
                WrappingRange { start: 1, .. } => {
                    return Some(get_nullable_type(cx, field_ty).unwrap());
                }
                WrappingRange { start, end } => {
                    unreachable!("Unhandled start and end range: ({}, {})", start, end)
                }
            };
        }
    }
    None
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    /// Check if the type is array and emit an unsafe type lint.
    fn check_for_array_ty(&mut self, sp: Span, ty: Ty<'tcx>) -> bool {
        if let ty::Array(..) = ty.kind() {
            self.emit_ffi_unsafe_type_lint(
                ty,
                sp,
                fluent::lint_improper_ctypes_array_reason,
                Some(fluent::lint_improper_ctypes_array_help),
            );
            true
        } else {
            false
        }
    }

    /// Checks if the given field's type is "ffi-safe".
    fn check_field_type_for_ffi(
        &self,
        cache: &mut FxHashSet<Ty<'tcx>>,
        field: &ty::FieldDef,
        substs: SubstsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        let field_ty = field.ty(self.cx.tcx, substs);
        if field_ty.has_opaque_types() {
            self.check_type_for_ffi(cache, field_ty)
        } else {
            let field_ty = self.cx.tcx.normalize_erasing_regions(self.cx.param_env, field_ty);
            self.check_type_for_ffi(cache, field_ty)
        }
    }

    /// Checks if the given `VariantDef`'s field types are "ffi-safe".
    fn check_variant_for_ffi(
        &self,
        cache: &mut FxHashSet<Ty<'tcx>>,
        ty: Ty<'tcx>,
        def: ty::AdtDef<'tcx>,
        variant: &ty::VariantDef,
        substs: SubstsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;

        let transparent_safety = def.repr().transparent().then(|| {
            // Can assume that at most one field is not a ZST, so only check
            // that field's type for FFI-safety.
            if let Some(field) = transparent_newtype_field(self.cx.tcx, variant) {
                return self.check_field_type_for_ffi(cache, field, substs);
            } else {
                // All fields are ZSTs; this means that the type should behave
                // like (), which is FFI-unsafe... except if all fields are PhantomData,
                // which is tested for below
                FfiUnsafe { ty, reason: fluent::lint_improper_ctypes_struct_zst, help: None }
            }
        });
        // We can't completely trust repr(C) markings; make sure the fields are
        // actually safe.
        let mut all_phantom = !variant.fields.is_empty();
        for field in &variant.fields {
            match self.check_field_type_for_ffi(cache, &field, substs) {
                FfiSafe => {
                    all_phantom = false;
                }
                FfiPhantom(..) if !def.repr().transparent() && def.is_enum() => {
                    return FfiUnsafe {
                        ty,
                        reason: fluent::lint_improper_ctypes_enum_phantomdata,
                        help: None,
                    };
                }
                FfiPhantom(..) => {}
                r => return transparent_safety.unwrap_or(r),
            }
        }

        if all_phantom { FfiPhantom(ty) } else { transparent_safety.unwrap_or(FfiSafe) }
    }

    /// Checks if the given type is "ffi-safe" (has a stable, well-defined
    /// representation which can be exported to C code).
    fn check_type_for_ffi(&self, cache: &mut FxHashSet<Ty<'tcx>>, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        use FfiResult::*;

        let tcx = self.cx.tcx;

        // Protect against infinite recursion, for example
        // `struct S(*mut S);`.
        // FIXME: A recursion limit is necessary as well, for irregular
        // recursive types.
        if !cache.insert(ty) {
            return FfiSafe;
        }

        match *ty.kind() {
            ty::Adt(def, substs) => {
                if def.is_box() && matches!(self.mode, CItemKind::Definition) {
                    if ty.boxed_ty().is_sized(tcx, self.cx.param_env) {
                        return FfiSafe;
                    } else {
                        return FfiUnsafe {
                            ty,
                            reason: fluent::lint_improper_ctypes_box,
                            help: None,
                        };
                    }
                }
                if def.is_phantom_data() {
                    return FfiPhantom(ty);
                }
                match def.adt_kind() {
                    AdtKind::Struct | AdtKind::Union => {
                        if !def.repr().c() && !def.repr().transparent() {
                            return FfiUnsafe {
                                ty,
                                reason: if def.is_struct() {
                                    fluent::lint_improper_ctypes_struct_layout_reason
                                } else {
                                    fluent::lint_improper_ctypes_union_layout_reason
                                },
                                help: if def.is_struct() {
                                    Some(fluent::lint_improper_ctypes_struct_layout_help)
                                } else {
                                    Some(fluent::lint_improper_ctypes_union_layout_help)
                                },
                            };
                        }

                        let is_non_exhaustive =
                            def.non_enum_variant().is_field_list_non_exhaustive();
                        if is_non_exhaustive && !def.did().is_local() {
                            return FfiUnsafe {
                                ty,
                                reason: if def.is_struct() {
                                    fluent::lint_improper_ctypes_struct_non_exhaustive
                                } else {
                                    fluent::lint_improper_ctypes_union_non_exhaustive
                                },
                                help: None,
                            };
                        }

                        if def.non_enum_variant().fields.is_empty() {
                            return FfiUnsafe {
                                ty,
                                reason: if def.is_struct() {
                                    fluent::lint_improper_ctypes_struct_fieldless_reason
                                } else {
                                    fluent::lint_improper_ctypes_union_fieldless_reason
                                },
                                help: if def.is_struct() {
                                    Some(fluent::lint_improper_ctypes_struct_fieldless_help)
                                } else {
                                    Some(fluent::lint_improper_ctypes_union_fieldless_help)
                                },
                            };
                        }

                        self.check_variant_for_ffi(cache, ty, def, def.non_enum_variant(), substs)
                    }
                    AdtKind::Enum => {
                        if def.variants().is_empty() {
                            // Empty enums are okay... although sort of useless.
                            return FfiSafe;
                        }

                        // Check for a repr() attribute to specify the size of the
                        // discriminant.
                        if !def.repr().c() && !def.repr().transparent() && def.repr().int.is_none()
                        {
                            // Special-case types like `Option<extern fn()>`.
                            if repr_nullable_ptr(self.cx, ty, self.mode).is_none() {
                                return FfiUnsafe {
                                    ty,
                                    reason: fluent::lint_improper_ctypes_enum_repr_reason,
                                    help: Some(fluent::lint_improper_ctypes_enum_repr_help),
                                };
                            }
                        }

                        if def.is_variant_list_non_exhaustive() && !def.did().is_local() {
                            return FfiUnsafe {
                                ty,
                                reason: fluent::lint_improper_ctypes_non_exhaustive,
                                help: None,
                            };
                        }

                        // Check the contained variants.
                        for variant in def.variants() {
                            let is_non_exhaustive = variant.is_field_list_non_exhaustive();
                            if is_non_exhaustive && !variant.def_id.is_local() {
                                return FfiUnsafe {
                                    ty,
                                    reason: fluent::lint_improper_ctypes_non_exhaustive_variant,
                                    help: None,
                                };
                            }

                            match self.check_variant_for_ffi(cache, ty, def, variant, substs) {
                                FfiSafe => (),
                                r => return r,
                            }
                        }

                        FfiSafe
                    }
                }
            }

            ty::Char => FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_char_reason,
                help: Some(fluent::lint_improper_ctypes_char_help),
            },

            ty::Int(ty::IntTy::I128) | ty::Uint(ty::UintTy::U128) => {
                FfiUnsafe { ty, reason: fluent::lint_improper_ctypes_128bit, help: None }
            }

            // Primitive types with a stable representation.
            ty::Bool | ty::Int(..) | ty::Uint(..) | ty::Float(..) | ty::Never => FfiSafe,

            ty::Slice(_) => FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_slice_reason,
                help: Some(fluent::lint_improper_ctypes_slice_help),
            },

            ty::Dynamic(..) => {
                FfiUnsafe { ty, reason: fluent::lint_improper_ctypes_dyn, help: None }
            }

            ty::Str => FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_str_reason,
                help: Some(fluent::lint_improper_ctypes_str_help),
            },

            ty::Tuple(..) => FfiUnsafe {
                ty,
                reason: fluent::lint_improper_ctypes_tuple_reason,
                help: Some(fluent::lint_improper_ctypes_tuple_help),
            },

            ty::RawPtr(ty::TypeAndMut { ty, .. }) | ty::Ref(_, ty, _)
                if {
                    matches!(self.mode, CItemKind::Definition)
                        && ty.is_sized(self.cx.tcx, self.cx.param_env)
                } =>
            {
                FfiSafe
            }

            ty::RawPtr(ty::TypeAndMut { ty, .. })
                if match ty.kind() {
                    ty::Tuple(tuple) => tuple.is_empty(),
                    _ => false,
                } =>
            {
                FfiSafe
            }

            ty::RawPtr(ty::TypeAndMut { ty, .. }) | ty::Ref(_, ty, _) => {
                self.check_type_for_ffi(cache, ty)
            }

            ty::Array(inner_ty, _) => self.check_type_for_ffi(cache, inner_ty),

            ty::FnPtr(sig) => {
                if self.is_internal_abi(sig.abi()) {
                    return FfiUnsafe {
                        ty,
                        reason: fluent::lint_improper_ctypes_fnptr_reason,
                        help: Some(fluent::lint_improper_ctypes_fnptr_help),
                    };
                }

                let sig = tcx.erase_late_bound_regions(sig);
                if !sig.output().is_unit() {
                    let r = self.check_type_for_ffi(cache, sig.output());
                    match r {
                        FfiSafe => {}
                        _ => {
                            return r;
                        }
                    }
                }
                for arg in sig.inputs() {
                    let r = self.check_type_for_ffi(cache, *arg);
                    match r {
                        FfiSafe => {}
                        _ => {
                            return r;
                        }
                    }
                }
                FfiSafe
            }

            ty::Foreign(..) => FfiSafe,

            // While opaque types are checked for earlier, if a projection in a struct field
            // normalizes to an opaque type, then it will reach this branch.
            ty::Alias(ty::Opaque, ..) => {
                FfiUnsafe { ty, reason: fluent::lint_improper_ctypes_opaque, help: None }
            }

            // `extern "C" fn` functions can have type parameters, which may or may not be FFI-safe,
            //  so they are currently ignored for the purposes of this lint.
            ty::Param(..) | ty::Alias(ty::Projection | ty::Inherent, ..)
                if matches!(self.mode, CItemKind::Definition) =>
            {
                FfiSafe
            }

            ty::Param(..)
            | ty::Alias(ty::Projection | ty::Inherent, ..)
            | ty::Infer(..)
            | ty::Bound(..)
            | ty::Error(_)
            | ty::Closure(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::GeneratorWitnessMIR(..)
            | ty::Placeholder(..)
            | ty::FnDef(..) => bug!("unexpected type in foreign function: {:?}", ty),
        }
    }

    fn emit_ffi_unsafe_type_lint(
        &mut self,
        ty: Ty<'tcx>,
        sp: Span,
        note: DiagnosticMessage,
        help: Option<DiagnosticMessage>,
    ) {
        let lint = match self.mode {
            CItemKind::Declaration => IMPROPER_CTYPES,
            CItemKind::Definition => IMPROPER_CTYPES_DEFINITIONS,
        };
        let desc = match self.mode {
            CItemKind::Declaration => "block",
            CItemKind::Definition => "fn",
        };
        let span_note = if let ty::Adt(def, _) = ty.kind()
            && let Some(sp) = self.cx.tcx.hir().span_if_local(def.did()) {
                Some(sp)
            } else {
                None
            };
        self.cx.emit_spanned_lint(
            lint,
            sp,
            ImproperCTypes { ty, desc, label: sp, help, note, span_note },
        );
    }

    fn check_for_opaque_ty(&mut self, sp: Span, ty: Ty<'tcx>) -> bool {
        struct ProhibitOpaqueTypes;
        impl<'tcx> ty::visit::TypeVisitor<TyCtxt<'tcx>> for ProhibitOpaqueTypes {
            type BreakTy = Ty<'tcx>;

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
                if !ty.has_opaque_types() {
                    return ControlFlow::Continue(());
                }

                if let ty::Alias(ty::Opaque, ..) = ty.kind() {
                    ControlFlow::Break(ty)
                } else {
                    ty.super_visit_with(self)
                }
            }
        }

        if let Some(ty) = self
            .cx
            .tcx
            .normalize_erasing_regions(self.cx.param_env, ty)
            .visit_with(&mut ProhibitOpaqueTypes)
            .break_value()
        {
            self.emit_ffi_unsafe_type_lint(ty, sp, fluent::lint_improper_ctypes_opaque, None);
            true
        } else {
            false
        }
    }

    fn check_type_for_ffi_and_report_errors(
        &mut self,
        sp: Span,
        ty: Ty<'tcx>,
        is_static: bool,
        is_return_type: bool,
    ) {
        // We have to check for opaque types before `normalize_erasing_regions`,
        // which will replace opaque types with their underlying concrete type.
        if self.check_for_opaque_ty(sp, ty) {
            // We've already emitted an error due to an opaque type.
            return;
        }

        // it is only OK to use this function because extern fns cannot have
        // any generic types right now:
        let ty = self.cx.tcx.normalize_erasing_regions(self.cx.param_env, ty);

        // C doesn't really support passing arrays by value - the only way to pass an array by value
        // is through a struct. So, first test that the top level isn't an array, and then
        // recursively check the types inside.
        if !is_static && self.check_for_array_ty(sp, ty) {
            return;
        }

        // Don't report FFI errors for unit return types. This check exists here, and not in
        // `check_foreign_fn` (where it would make more sense) so that normalization has definitely
        // happened.
        if is_return_type && ty.is_unit() {
            return;
        }

        match self.check_type_for_ffi(&mut FxHashSet::default(), ty) {
            FfiResult::FfiSafe => {}
            FfiResult::FfiPhantom(ty) => {
                self.emit_ffi_unsafe_type_lint(
                    ty,
                    sp,
                    fluent::lint_improper_ctypes_only_phantomdata,
                    None,
                );
            }
            // If `ty` is a `repr(transparent)` newtype, and the non-zero-sized type is a generic
            // argument, which after substitution, is `()`, then this branch can be hit.
            FfiResult::FfiUnsafe { ty, .. } if is_return_type && ty.is_unit() => {}
            FfiResult::FfiUnsafe { ty, reason, help } => {
                self.emit_ffi_unsafe_type_lint(ty, sp, reason, help);
            }
        }
    }

    fn check_foreign_fn(&mut self, def_id: LocalDefId, decl: &hir::FnDecl<'_>) {
        let sig = self.cx.tcx.fn_sig(def_id).subst_identity();
        let sig = self.cx.tcx.erase_late_bound_regions(sig);

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            self.check_type_for_ffi_and_report_errors(input_hir.span, *input_ty, false, false);
        }

        if let hir::FnRetTy::Return(ref ret_hir) = decl.output {
            let ret_ty = sig.output();
            self.check_type_for_ffi_and_report_errors(ret_hir.span, ret_ty, false, true);
        }
    }

    fn check_foreign_static(&mut self, id: hir::OwnerId, span: Span) {
        let ty = self.cx.tcx.type_of(id).subst_identity();
        self.check_type_for_ffi_and_report_errors(span, ty, true, false);
    }

    fn is_internal_abi(&self, abi: SpecAbi) -> bool {
        matches!(
            abi,
            SpecAbi::Rust | SpecAbi::RustCall | SpecAbi::RustIntrinsic | SpecAbi::PlatformIntrinsic
        )
    }
}

impl<'tcx> LateLintPass<'tcx> for ImproperCTypesDeclarations {
    fn check_foreign_item(&mut self, cx: &LateContext<'_>, it: &hir::ForeignItem<'_>) {
        let mut vis = ImproperCTypesVisitor { cx, mode: CItemKind::Declaration };
        let abi = cx.tcx.hir().get_foreign_abi(it.hir_id());

        if !vis.is_internal_abi(abi) {
            match it.kind {
                hir::ForeignItemKind::Fn(ref decl, _, _) => {
                    vis.check_foreign_fn(it.owner_id.def_id, decl);
                }
                hir::ForeignItemKind::Static(ref ty, _) => {
                    vis.check_foreign_static(it.owner_id, ty.span);
                }
                hir::ForeignItemKind::Type => (),
            }
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for ImproperCTypesDefinitions {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: hir::intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'_>,
        _: &'tcx hir::Body<'_>,
        _: Span,
        id: LocalDefId,
    ) {
        use hir::intravisit::FnKind;

        let abi = match kind {
            FnKind::ItemFn(_, _, header, ..) => header.abi,
            FnKind::Method(_, sig, ..) => sig.header.abi,
            _ => return,
        };

        let mut vis = ImproperCTypesVisitor { cx, mode: CItemKind::Definition };
        if !vis.is_internal_abi(abi) {
            vis.check_foreign_fn(id, decl);
        }
    }
}

declare_lint_pass!(VariantSizeDifferences => [VARIANT_SIZE_DIFFERENCES]);

impl<'tcx> LateLintPass<'tcx> for VariantSizeDifferences {
    fn check_item(&mut self, cx: &LateContext<'_>, it: &hir::Item<'_>) {
        if let hir::ItemKind::Enum(ref enum_definition, _) = it.kind {
            let t = cx.tcx.type_of(it.owner_id).subst_identity();
            let ty = cx.tcx.erase_regions(t);
            let Ok(layout) = cx.layout_of(ty) else { return };
            let Variants::Multiple {
                    tag_encoding: TagEncoding::Direct, tag, ref variants, ..
                } = &layout.variants else {
                return
            };

            let tag_size = tag.size(&cx.tcx).bytes();

            debug!(
                "enum `{}` is {} bytes large with layout:\n{:#?}",
                t,
                layout.size.bytes(),
                layout
            );

            let (largest, slargest, largest_index) = iter::zip(enum_definition.variants, variants)
                .map(|(variant, variant_layout)| {
                    // Subtract the size of the enum tag.
                    let bytes = variant_layout.size.bytes().saturating_sub(tag_size);

                    debug!("- variant `{}` is {} bytes large", variant.ident, bytes);
                    bytes
                })
                .enumerate()
                .fold((0, 0, 0), |(l, s, li), (idx, size)| {
                    if size > l {
                        (size, l, idx)
                    } else if size > s {
                        (l, size, li)
                    } else {
                        (l, s, li)
                    }
                });

            // We only warn if the largest variant is at least thrice as large as
            // the second-largest.
            if largest > slargest * 3 && slargest > 0 {
                cx.emit_spanned_lint(
                    VARIANT_SIZE_DIFFERENCES,
                    enum_definition.variants[largest_index].span,
                    VariantSizeDifferencesDiag { largest },
                );
            }
        }
    }
}

declare_lint! {
    /// The `invalid_atomic_ordering` lint detects passing an `Ordering`
    /// to an atomic operation that does not support that ordering.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// # use core::sync::atomic::{AtomicU8, Ordering};
    /// let atom = AtomicU8::new(0);
    /// let value = atom.load(Ordering::Release);
    /// # let _ = value;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Some atomic operations are only supported for a subset of the
    /// `atomic::Ordering` variants. Passing an unsupported variant will cause
    /// an unconditional panic at runtime, which is detected by this lint.
    ///
    /// This lint will trigger in the following cases: (where `AtomicType` is an
    /// atomic type from `core::sync::atomic`, such as `AtomicBool`,
    /// `AtomicPtr`, `AtomicUsize`, or any of the other integer atomics).
    ///
    /// - Passing `Ordering::Acquire` or `Ordering::AcqRel` to
    ///   `AtomicType::store`.
    ///
    /// - Passing `Ordering::Release` or `Ordering::AcqRel` to
    ///   `AtomicType::load`.
    ///
    /// - Passing `Ordering::Relaxed` to `core::sync::atomic::fence` or
    ///   `core::sync::atomic::compiler_fence`.
    ///
    /// - Passing `Ordering::Release` or `Ordering::AcqRel` as the failure
    ///   ordering for any of `AtomicType::compare_exchange`,
    ///   `AtomicType::compare_exchange_weak`, or `AtomicType::fetch_update`.
    INVALID_ATOMIC_ORDERING,
    Deny,
    "usage of invalid atomic ordering in atomic operations and memory fences"
}

declare_lint_pass!(InvalidAtomicOrdering => [INVALID_ATOMIC_ORDERING]);

impl InvalidAtomicOrdering {
    fn inherent_atomic_method_call<'hir>(
        cx: &LateContext<'_>,
        expr: &Expr<'hir>,
        recognized_names: &[Symbol], // used for fast path calculation
    ) -> Option<(Symbol, &'hir [Expr<'hir>])> {
        const ATOMIC_TYPES: &[Symbol] = &[
            sym::AtomicBool,
            sym::AtomicPtr,
            sym::AtomicUsize,
            sym::AtomicU8,
            sym::AtomicU16,
            sym::AtomicU32,
            sym::AtomicU64,
            sym::AtomicU128,
            sym::AtomicIsize,
            sym::AtomicI8,
            sym::AtomicI16,
            sym::AtomicI32,
            sym::AtomicI64,
            sym::AtomicI128,
        ];
        if let ExprKind::MethodCall(ref method_path, _, args, _) = &expr.kind
            && recognized_names.contains(&method_path.ident.name)
            && let Some(m_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
            && let Some(impl_did) = cx.tcx.impl_of_method(m_def_id)
            && let Some(adt) = cx.tcx.type_of(impl_did).subst_identity().ty_adt_def()
            // skip extension traits, only lint functions from the standard library
            && cx.tcx.trait_id_of_impl(impl_did).is_none()
            && let parent = cx.tcx.parent(adt.did())
            && cx.tcx.is_diagnostic_item(sym::atomic_mod, parent)
            && ATOMIC_TYPES.contains(&cx.tcx.item_name(adt.did()))
        {
            return Some((method_path.ident.name, args));
        }
        None
    }

    fn match_ordering(cx: &LateContext<'_>, ord_arg: &Expr<'_>) -> Option<Symbol> {
        let ExprKind::Path(ref ord_qpath) = ord_arg.kind else { return None };
        let did = cx.qpath_res(ord_qpath, ord_arg.hir_id).opt_def_id()?;
        let tcx = cx.tcx;
        let atomic_ordering = tcx.get_diagnostic_item(sym::Ordering);
        let name = tcx.item_name(did);
        let parent = tcx.parent(did);
        [sym::Relaxed, sym::Release, sym::Acquire, sym::AcqRel, sym::SeqCst].into_iter().find(
            |&ordering| {
                name == ordering
                    && (Some(parent) == atomic_ordering
                            // needed in case this is a ctor, not a variant
                            || tcx.opt_parent(parent) == atomic_ordering)
            },
        )
    }

    fn check_atomic_load_store(cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let Some((method, args)) = Self::inherent_atomic_method_call(cx, expr, &[sym::load, sym::store])
            && let Some((ordering_arg, invalid_ordering)) = match method {
                sym::load => Some((&args[0], sym::Release)),
                sym::store => Some((&args[1], sym::Acquire)),
                _ => None,
            }
            && let Some(ordering) = Self::match_ordering(cx, ordering_arg)
            && (ordering == invalid_ordering || ordering == sym::AcqRel)
        {
            if method == sym::load {
                cx.emit_spanned_lint(INVALID_ATOMIC_ORDERING, ordering_arg.span, AtomicOrderingLoad);
            } else {
                cx.emit_spanned_lint(INVALID_ATOMIC_ORDERING, ordering_arg.span, AtomicOrderingStore);
            };
        }
    }

    fn check_memory_fence(cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::Call(ref func, ref args) = expr.kind
            && let ExprKind::Path(ref func_qpath) = func.kind
            && let Some(def_id) = cx.qpath_res(func_qpath, func.hir_id).opt_def_id()
            && matches!(cx.tcx.get_diagnostic_name(def_id), Some(sym::fence | sym::compiler_fence))
            && Self::match_ordering(cx, &args[0]) == Some(sym::Relaxed)
        {
            cx.emit_spanned_lint(INVALID_ATOMIC_ORDERING, args[0].span, AtomicOrderingFence);
        }
    }

    fn check_atomic_compare_exchange(cx: &LateContext<'_>, expr: &Expr<'_>) {
        let Some((method, args)) = Self::inherent_atomic_method_call(cx, expr, &[sym::fetch_update, sym::compare_exchange, sym::compare_exchange_weak])
            else {return };

        let fail_order_arg = match method {
            sym::fetch_update => &args[1],
            sym::compare_exchange | sym::compare_exchange_weak => &args[3],
            _ => return,
        };

        let Some(fail_ordering) = Self::match_ordering(cx, fail_order_arg) else { return };

        if matches!(fail_ordering, sym::Release | sym::AcqRel) {
            cx.emit_spanned_lint(
                INVALID_ATOMIC_ORDERING,
                fail_order_arg.span,
                InvalidAtomicOrderingDiag { method, fail_order_arg_span: fail_order_arg.span },
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for InvalidAtomicOrdering {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        Self::check_atomic_load_store(cx, expr);
        Self::check_memory_fence(cx, expr);
        Self::check_atomic_compare_exchange(cx, expr);
    }
}
