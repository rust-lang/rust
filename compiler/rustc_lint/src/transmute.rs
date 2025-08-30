use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{self as hir};
use rustc_macros::LintDiagnostic;
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::sym;

use crate::lints::{IntegerToPtrTransmutes, IntegerToPtrTransmutesSuggestion};
use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `ptr_to_integer_transmute_in_consts` lint detects pointer to integer
    /// transmute in const functions and associated constants.
    ///
    /// ### Example
    ///
    /// ```rust
    /// const fn foo(ptr: *const u8) -> usize {
    ///    unsafe {
    ///        std::mem::transmute::<*const u8, usize>(ptr)
    ///    }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Transmuting pointers to integers in a `const` context is undefined behavior.
    /// Any attempt to use the resulting integer will abort const-evaluation.
    ///
    /// But sometimes the compiler might not emit an error for pointer to integer transmutes
    /// inside const functions and associated consts because they are evaluated only when referenced.
    /// Therefore, this lint serves as an extra layer of defense to prevent any undefined behavior
    /// from compiling without any warnings or errors.
    ///
    /// See [std::mem::transmute] in the reference for more details.
    ///
    /// [std::mem::transmute]: https://doc.rust-lang.org/std/mem/fn.transmute.html
    pub PTR_TO_INTEGER_TRANSMUTE_IN_CONSTS,
    Warn,
    "detects pointer to integer transmutes in const functions and associated constants",
}

declare_lint! {
    /// The `unnecessary_transmutes` lint detects transmutations that have safer alternatives.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn bytes_at_home(x: [u8; 4]) -> u32 {
    ///   unsafe { std::mem::transmute(x) }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Using an explicit method is preferable over calls to
    /// [`transmute`](https://doc.rust-lang.org/std/mem/fn.transmute.html) as
    /// they more clearly communicate the intent, are easier to review, and
    /// are less likely to accidentally result in unsoundness.
    pub UNNECESSARY_TRANSMUTES,
    Warn,
    "detects transmutes that can also be achieved by other operations"
}

declare_lint! {
    /// The `integer_to_ptr_transmutes` lint detects integer to pointer
    /// transmutes where the resulting pointers are undefined behavior to dereference.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn foo(a: usize) -> *const u8 {
    ///    unsafe {
    ///        std::mem::transmute::<usize, *const u8>(a)
    ///    }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Any attempt to use the resulting pointers are undefined behavior as the resulting
    /// pointers won't have any provenance.
    ///
    /// Alternatively, [`std::ptr::with_exposed_provenance`] should be used, as they do not
    /// carry the provenance requirement. If wanting to create pointers without provenance
    /// [`std::ptr::without_provenance`] should be used instead.
    ///
    /// See [`std::mem::transmute`] in the reference for more details.
    ///
    /// [`std::mem::transmute`]: https://doc.rust-lang.org/std/mem/fn.transmute.html
    /// [`std::ptr::with_exposed_provenance`]: https://doc.rust-lang.org/std/ptr/fn.with_exposed_provenance.html
    /// [`std::ptr::without_provenance`]: https://doc.rust-lang.org/std/ptr/fn.without_provenance.html
    pub INTEGER_TO_PTR_TRANSMUTES,
    Warn,
    "detects integer to pointer transmutes",
}

pub(crate) struct CheckTransmutes;

impl_lint_pass!(CheckTransmutes => [PTR_TO_INTEGER_TRANSMUTE_IN_CONSTS, UNNECESSARY_TRANSMUTES, INTEGER_TO_PTR_TRANSMUTES]);

impl<'tcx> LateLintPass<'tcx> for CheckTransmutes {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        let hir::ExprKind::Call(callee, [arg]) = expr.kind else {
            return;
        };
        let hir::ExprKind::Path(qpath) = callee.kind else {
            return;
        };
        let Res::Def(DefKind::Fn, def_id) = cx.qpath_res(&qpath, callee.hir_id) else {
            return;
        };
        if !cx.tcx.is_intrinsic(def_id, sym::transmute) {
            return;
        };
        let body_owner_def_id = cx.tcx.hir_enclosing_body_owner(expr.hir_id);
        let const_context = cx.tcx.hir_body_const_context(body_owner_def_id);
        let args = cx.typeck_results().node_args(callee.hir_id);

        let src = args.type_at(0);
        let dst = args.type_at(1);

        check_ptr_transmute_in_const(cx, expr, body_owner_def_id, const_context, src, dst);
        check_unnecessary_transmute(cx, expr, callee, arg, const_context, src, dst);
        check_int_to_ptr_transmute(cx, expr, arg, src, dst);
    }
}

/// Check for transmutes from integer to pointers (*const/*mut and &/&mut).
///
/// Using the resulting pointers would be undefined behavior.
fn check_int_to_ptr_transmute<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    arg: &'tcx hir::Expr<'tcx>,
    src: Ty<'tcx>,
    dst: Ty<'tcx>,
) {
    if !matches!(src.kind(), ty::Uint(_) | ty::Int(_)) {
        return;
    }
    let (ty::Ref(_, inner_ty, mutbl) | ty::RawPtr(inner_ty, mutbl)) = dst.kind() else {
        return;
    };
    // bail-out if the argument is literal 0 as we have other lints for those cases
    if matches!(arg.kind, hir::ExprKind::Lit(hir::Lit { node: LitKind::Int(v, _), .. }) if v == 0) {
        return;
    }
    // bail-out if the inner type is a ZST
    let Ok(layout_inner_ty) = cx.tcx.layout_of(cx.typing_env().as_query_input(*inner_ty)) else {
        return;
    };
    if layout_inner_ty.is_1zst() {
        return;
    }

    let suffix = if mutbl.is_mut() { "_mut" } else { "" };
    cx.tcx.emit_node_span_lint(
        INTEGER_TO_PTR_TRANSMUTES,
        expr.hir_id,
        expr.span,
        IntegerToPtrTransmutes {
            suggestion: if layout_inner_ty.is_sized() {
                Some(if dst.is_ref() {
                    IntegerToPtrTransmutesSuggestion::ToRef {
                        dst: *inner_ty,
                        suffix,
                        ref_mutbl: mutbl.prefix_str(),
                        start_call: expr.span.shrink_to_lo().until(arg.span),
                    }
                } else {
                    IntegerToPtrTransmutesSuggestion::ToPtr {
                        dst: *inner_ty,
                        suffix,
                        start_call: expr.span.shrink_to_lo().until(arg.span),
                    }
                })
            } else {
                // We can't suggest using `with_exposed_provenance` for unsized type
                // so don't suggest anything.
                None
            },
        },
    );
}

/// Check for transmutes that exhibit undefined behavior.
/// For example, transmuting pointers to integers in a const context.
///
/// Why do we consider const functions and associated constants only?
///
/// Generally, undefined behavior in const items are handled by the evaluator.
/// But, const functions and associated constants are evaluated only when referenced.
/// This can result in undefined behavior in a library going unnoticed until
/// the function or constant is actually used.
///
/// Therefore, we only consider const functions and associated constants here and leave
/// other const items to be handled by the evaluator.
fn check_ptr_transmute_in_const<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    body_owner_def_id: LocalDefId,
    const_context: Option<hir::ConstContext>,
    src: Ty<'tcx>,
    dst: Ty<'tcx>,
) {
    if matches!(const_context, Some(hir::ConstContext::ConstFn))
        || matches!(cx.tcx.def_kind(body_owner_def_id), DefKind::AssocConst)
    {
        if src.is_raw_ptr() && dst.is_integral() {
            cx.tcx.emit_node_span_lint(
                PTR_TO_INTEGER_TRANSMUTE_IN_CONSTS,
                expr.hir_id,
                expr.span,
                UndefinedTransmuteLint,
            );
        }
    }
}

/// Check for transmutes that overlap with stdlib methods.
/// For example, transmuting `[u8; 4]` to `u32`.
///
/// We chose not to lint u8 -> bool transmutes, see #140431.
fn check_unnecessary_transmute<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    callee: &'tcx hir::Expr<'tcx>,
    arg: &'tcx hir::Expr<'tcx>,
    const_context: Option<hir::ConstContext>,
    src: Ty<'tcx>,
    dst: Ty<'tcx>,
) {
    let callee_span = callee.span.find_ancestor_inside(expr.span).unwrap_or(callee.span);
    let (sugg, help) = match (src.kind(), dst.kind()) {
        // dont check the length; transmute does that for us.
        // [u8; _] => primitive
        (ty::Array(t, _), ty::Uint(_) | ty::Float(_) | ty::Int(_))
            if *t.kind() == ty::Uint(ty::UintTy::U8) =>
        {
            (
                Some(vec![(callee_span, format!("{dst}::from_ne_bytes"))]),
                Some(
                    "there's also `from_le_bytes` and `from_be_bytes` if you expect a particular byte order",
                ),
            )
        }
        // primitive => [u8; _]
        (ty::Uint(_) | ty::Float(_) | ty::Int(_), ty::Array(t, _))
            if *t.kind() == ty::Uint(ty::UintTy::U8) =>
        {
            (
                Some(vec![(callee_span, format!("{src}::to_ne_bytes"))]),
                Some(
                    "there's also `to_le_bytes` and `to_be_bytes` if you expect a particular byte order",
                ),
            )
        }
        // char → u32
        (ty::Char, ty::Uint(ty::UintTy::U32)) => {
            (Some(vec![(callee_span, "u32::from".to_string())]), None)
        }
        // char (→ u32) → i32
        (ty::Char, ty::Int(ty::IntTy::I32)) => (
            Some(vec![
                (callee_span, "u32::from".to_string()),
                (expr.span.shrink_to_hi(), ".cast_signed()".to_string()),
            ]),
            None,
        ),
        // u32 → char
        (ty::Uint(ty::UintTy::U32), ty::Char) => (
            Some(vec![(callee_span, "char::from_u32_unchecked".to_string())]),
            Some("consider using `char::from_u32(…).unwrap()`"),
        ),
        // i32 → char
        (ty::Int(ty::IntTy::I32), ty::Char) => (
            Some(vec![
                (callee_span, "char::from_u32_unchecked(i32::cast_unsigned".to_string()),
                (expr.span.shrink_to_hi(), ")".to_string()),
            ]),
            Some("consider using `char::from_u32(i32::cast_unsigned(…)).unwrap()`"),
        ),
        // uNN → iNN
        (ty::Uint(_), ty::Int(_)) => {
            (Some(vec![(callee_span, format!("{src}::cast_signed"))]), None)
        }
        // iNN → uNN
        (ty::Int(_), ty::Uint(_)) => {
            (Some(vec![(callee_span, format!("{src}::cast_unsigned"))]), None)
        }
        // fNN → usize, isize
        (ty::Float(_), ty::Uint(ty::UintTy::Usize) | ty::Int(ty::IntTy::Isize)) => (
            Some(vec![
                (callee_span, format!("{src}::to_bits")),
                (expr.span.shrink_to_hi(), format!(" as {dst}")),
            ]),
            None,
        ),
        // fNN (→ uNN) → iNN
        (ty::Float(_), ty::Int(..)) => (
            Some(vec![
                (callee_span, format!("{src}::to_bits")),
                (expr.span.shrink_to_hi(), ".cast_signed()".to_string()),
            ]),
            None,
        ),
        // fNN → uNN
        (ty::Float(_), ty::Uint(..)) => {
            (Some(vec![(callee_span, format!("{src}::to_bits"))]), None)
        }
        // xsize → fNN
        (ty::Uint(ty::UintTy::Usize) | ty::Int(ty::IntTy::Isize), ty::Float(_)) => (
            Some(vec![
                (callee_span, format!("{dst}::from_bits")),
                (arg.span.shrink_to_hi(), " as _".to_string()),
            ]),
            None,
        ),
        // iNN (→ uNN) → fNN
        (ty::Int(_), ty::Float(_)) => (
            Some(vec![
                (callee_span, format!("{dst}::from_bits({src}::cast_unsigned")),
                (expr.span.shrink_to_hi(), ")".to_string()),
            ]),
            None,
        ),
        // uNN → fNN
        (ty::Uint(_), ty::Float(_)) => {
            (Some(vec![(callee_span, format!("{dst}::from_bits"))]), None)
        }
        // bool → x8 in const context since `From::from` is not const yet
        // FIXME: Consider arg expr's precedence to avoid parentheses.
        // FIXME(const_traits): Remove this when `From::from` is constified.
        (ty::Bool, ty::Int(..) | ty::Uint(..)) if const_context.is_some() => (
            Some(vec![
                (callee_span, "".to_string()),
                (expr.span.shrink_to_hi(), format!(" as {dst}")),
            ]),
            None,
        ),
        // bool → x8 using `x8::from`
        (ty::Bool, ty::Int(..) | ty::Uint(..)) => {
            (Some(vec![(callee_span, format!("{dst}::from"))]), None)
        }
        _ => return,
    };

    cx.tcx.node_span_lint(UNNECESSARY_TRANSMUTES, expr.hir_id, expr.span, |diag| {
        diag.primary_message("unnecessary transmute");
        if let Some(sugg) = sugg {
            diag.multipart_suggestion("replace this with", sugg, Applicability::MachineApplicable);
        }
        if let Some(help) = help {
            diag.help(help);
        }
    });
}

#[derive(LintDiagnostic)]
#[diag(lint_undefined_transmute)]
#[note]
#[note(lint_note2)]
#[help]
pub(crate) struct UndefinedTransmuteLint;
