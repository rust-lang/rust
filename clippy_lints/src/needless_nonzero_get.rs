use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::Msrv;
use clippy_utils::ty::implements_trait;
use clippy_utils::{binop_traits, is_from_proc_macro, span_contains_comment, sym};
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{AssignOpKind, BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::impl_lint_pass;
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `NonZero::get` calls that are immediately followed by a method or operator
    /// which `NonZero` provides itself, with the same return type.
    ///
    /// ### Why is this bad?
    /// The `get` call adds nothing but noise, as the method could be called on the
    /// `NonZero` value directly.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::num::NonZero;
    /// # let nz = NonZero::new(1u32).unwrap();
    /// let _ = nz.get().leading_zeros();
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::num::NonZero;
    /// # let nz = NonZero::new(1u32).unwrap();
    /// let _ = nz.leading_zeros();
    /// ```
    ///
    /// The lint also handles division and remainder operators:
    /// ```no_run
    /// # use std::num::NonZero;
    /// # let nz = NonZero::new(2u32).unwrap();
    /// let _ = 4 / nz.get();
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::num::NonZero;
    /// # let nz = NonZero::new(2u32).unwrap();
    /// let _ = 4 / nz;
    /// ```
    #[clippy::version = "1.99.0"]
    pub NEEDLESS_NONZERO_GET,
    complexity,
    "unnecessary `NonZero::get` call"
}

impl_lint_pass!(NeedlessNonzeroGet => [NEEDLESS_NONZERO_GET]);

pub struct NeedlessNonzeroGet {
    msrv: Msrv,
}

impl NeedlessNonzeroGet {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv.into() }
    }
}

impl<'tcx> LateLintPass<'tcx> for NeedlessNonzeroGet {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        match expr.kind {
            // `<recv>.get().<method>()`
            ExprKind::MethodCall(method, get_call, [], _)
                if let Some((recv, nz_ty, nonzero_did, get_span)) = nonzero_get(cx, get_call)
                    // The outer call must resolve to an inherent method. A trait method of the same
                    // name could resolve differently once the receiver becomes a `NonZero`.
                    && let Some(method_did) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
                    && cx.tcx.trait_of_assoc(method_did).is_none()
                    // Dropping `get` must leave the expression's type unchanged. Methods whose
                    // `NonZero` version returns a `NonZero` deliberately do not match.
                    && let ret_ty = cx.typeck_results().expr_ty(expr)
                    && has_matching_method(cx, nonzero_did, nz_ty, method.ident.name, ret_ty, self.msrv) =>
            {
                emit_unnecessary_get(cx, expr, get_call, recv, get_span, method.ident.name.as_str());
            },

            // `<lhs> / <recv>.get()` and `<lhs> % <recv>.get()`
            ExprKind::Binary(op, lhs, get_call)
                if matches!(op.node, BinOpKind::Div | BinOpKind::Rem)
                    && let Some((recv, nz_ty, _, get_span)) = nonzero_get(cx, get_call)
                    && let Some((trait_lang_item, _)) = binop_traits(op.node)
                    && let Some(trait_id) = cx.tcx.lang_items().get(trait_lang_item)
                    && has_nonzero_operator(cx, lhs, recv, nz_ty, trait_id, self.msrv) =>
            {
                emit_unnecessary_get(cx, expr, get_call, recv, get_span, op.node.as_str());
            },

            // `<lhs> /= <recv>.get()` and `<lhs> %= <recv>.get()`
            ExprKind::AssignOp(op, lhs, get_call)
                if matches!(op.node, AssignOpKind::DivAssign | AssignOpKind::RemAssign)
                    && let Some((recv, nz_ty, _, get_span)) = nonzero_get(cx, get_call)
                    && let Some((_, trait_lang_item)) = binop_traits(op.node.into())
                    && let Some(trait_id) = cx.tcx.lang_items().get(trait_lang_item)
                    && has_nonzero_operator(cx, lhs, recv, nz_ty, trait_id, self.msrv) =>
            {
                emit_unnecessary_get(cx, expr, get_call, recv, get_span, op.node.as_str());
            },

            _ => {},
        }
    }
}

/// Returns the receiver, its `NonZero` type and definition, and the `get` identifier span when
/// `expr` is an inherent `NonZero::get()` call.
fn nonzero_get<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
) -> Option<(&'tcx Expr<'tcx>, Ty<'tcx>, DefId, Span)> {
    let ExprKind::MethodCall(get, recv, [], get_span) = expr.kind else {
        return None;
    };
    if get.ident.name != sym::get {
        return None;
    }
    let get_did = cx.typeck_results().type_dependent_def_id(expr.hir_id)?;
    if cx.tcx.trait_of_assoc(get_did).is_some() {
        return None;
    }
    let nz_ty = cx.typeck_results().expr_ty_adjusted(recv);
    let ty::Adt(adt, _) = nz_ty.kind() else {
        return None;
    };
    if !cx.tcx.is_diagnostic_item(sym::NonZero, adt.did()) {
        return None;
    }

    Some((recv, nz_ty, adt.did(), get_span))
}

fn emit_unnecessary_get<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    get_call: &Expr<'tcx>,
    recv: &Expr<'tcx>,
    get_span: Span,
    operation: &str,
) {
    // Removing `.get()` means editing the span between the receiver and the surrounding
    // expression, which is only meaningful when both are written out in the same context.
    if expr.span.from_expansion()
        || !recv.span.eq_ctxt(expr.span)
        || !get_call.span.eq_ctxt(expr.span)
        || is_from_proc_macro(cx, expr)
    {
        return;
    }

    span_lint_and_then(
        cx,
        NEEDLESS_NONZERO_GET,
        get_span,
        format!("unnecessary `get` before `{operation}`"),
        |diag| {
            // Covers `.get()` including the dot and any whitespace before it.
            let removal_span = get_call.span.with_lo(recv.span.hi());
            let applicability = if span_contains_comment(cx, removal_span) {
                Applicability::MaybeIncorrect
            } else {
                Applicability::MachineApplicable
            };
            diag.span_suggestion_verbose(removal_span, "remove this", "", applicability);
        },
    );
}

/// Checks whether replacing the right-hand primitive operand with its `NonZero` receiver resolves
/// to an unsigned standard-library operator implementation that is usable under `msrv`. In a
/// `const` context that means const-stable, which the `NonZero` operator implementations are not.
fn has_nonzero_operator<'tcx>(
    cx: &LateContext<'tcx>,
    lhs: &'tcx Expr<'tcx>,
    recv: &'tcx Expr<'tcx>,
    nz_ty: Ty<'tcx>,
    trait_id: DefId,
    msrv: Msrv,
) -> bool {
    let lhs_ty = cx.typeck_results().expr_ty(lhs);
    let ty::Adt(_, args) = nz_ty.kind() else {
        return false;
    };
    let inner_ty = args.type_at(0);

    // Primitive operators forward some reference operands, whereas the `NonZero` implementations
    // do not. Require the exact written operand types so the replacement is guaranteed to resolve.
    lhs_ty == inner_ty
        && cx.typeck_results().expr_ty(recv) == nz_ty
        && matches!(lhs_ty.kind(), ty::Uint(_))
        && implements_trait(cx, lhs_ty, trait_id, &[nz_ty.into()])
        && cx.tcx.non_blanket_impls_for_ty(trait_id, lhs_ty).any(|impl_id| {
            let trait_ref = cx.tcx.impl_trait_ref(impl_id).instantiate_identity().skip_norm_wip();
            trait_ref.args.type_at(1) == nz_ty && msrv.is_stable_or_const_stable(cx, impl_id)
        })
}

/// Checks whether `nz_ty` has an inherent method `name` taking nothing but `self` and returning
/// exactly `ret_ty`, which is usable under `msrv` (const-stable when in a `const` context, stable
/// otherwise).
fn has_matching_method<'tcx>(
    cx: &LateContext<'tcx>,
    nonzero_did: DefId,
    nz_ty: Ty<'tcx>,
    name: Symbol,
    ret_ty: Ty<'tcx>,
    msrv: Msrv,
) -> bool {
    cx.tcx.inherent_impls(nonzero_did).iter().any(|&impl_did| {
        // The integer methods live in concrete `impl NonZero<u32>`-style blocks, so their
        // signatures need no instantiation. Restricting to the impl matching the receiver also
        // keeps signed-only methods off unsigned `NonZero`s and vice versa.
        cx.tcx.type_of(impl_did).instantiate_identity().skip_norm_wip() == nz_ty
            && cx
                .tcx
                .associated_items(impl_did)
                .filter_by_name_unhygienic(name)
                .any(|item| {
                    item.is_fn() && {
                        let sig = cx.tcx.fn_sig(item.def_id).instantiate_identity().skip_binder();
                        sig.inputs().len() == 1
                            && sig.output() == ret_ty
                            && msrv.is_stable_or_const_stable(cx, item.def_id)
                    }
                })
    })
}
