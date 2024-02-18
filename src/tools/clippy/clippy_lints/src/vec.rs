use std::collections::BTreeMap;
use std::ops::ControlFlow;

use clippy_config::msrvs::{self, Msrv};
use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_copy;
use clippy_utils::visitors::for_each_local_use_after_expr;
use clippy_utils::{get_parent_expr, higher, is_trait_method};
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, HirId, Mutability, Node, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_middle::ty::layout::LayoutOf;
use rustc_session::impl_lint_pass;
use rustc_span::{sym, DesugaringKind, Span};

#[expect(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct UselessVec {
    pub too_large_for_stack: u64,
    pub msrv: Msrv,
    pub span_to_lint_map: BTreeMap<Span, Option<(HirId, SuggestedType, String, Applicability)>>,
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `vec![..]` when using `[..]` would
    /// be possible.
    ///
    /// ### Why is this bad?
    /// This is less efficient.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo(_x: &[u8]) {}
    ///
    /// foo(&vec![1, 2]);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # fn foo(_x: &[u8]) {}
    /// foo(&[1, 2]);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub USELESS_VEC,
    perf,
    "useless `vec!`"
}

impl_lint_pass!(UselessVec => [USELESS_VEC]);

fn adjusts_to_slice(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    matches!(cx.typeck_results().expr_ty_adjusted(e).kind(), ty::Ref(_, ty, _) if ty.is_slice())
}

/// Checks if the given expression is a method call to a `Vec` method
/// that also exists on slices. If this returns true, it means that
/// this expression does not actually require a `Vec` and could just work with an array.
pub fn is_allowed_vec_method(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    const ALLOWED_METHOD_NAMES: &[&str] = &["len", "as_ptr", "is_empty"];

    if let ExprKind::MethodCall(path, ..) = e.kind {
        ALLOWED_METHOD_NAMES.contains(&path.ident.name.as_str())
    } else {
        is_trait_method(cx, e, sym::IntoIterator)
    }
}

impl<'tcx> LateLintPass<'tcx> for UselessVec {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let Some(vec_args) = higher::VecArgs::hir(cx, expr.peel_borrows()) {
            // search for `let foo = vec![_]` expressions where all uses of `foo`
            // adjust to slices or call a method that exist on slices (e.g. len)
            if let Node::Local(local) = cx.tcx.parent_hir_node(expr.hir_id)
                // for now ignore locals with type annotations.
                // this is to avoid compile errors when doing the suggestion here: let _: Vec<_> = vec![..];
                && local.ty.is_none()
                && let PatKind::Binding(_, id, ..) = local.pat.kind
            {
                let only_slice_uses = for_each_local_use_after_expr(cx, id, expr.hir_id, |expr| {
                    // allow indexing into a vec and some set of allowed method calls that exist on slices, too
                    if let Some(parent) = get_parent_expr(cx, expr)
                        && (adjusts_to_slice(cx, expr)
                            || matches!(parent.kind, ExprKind::Index(..))
                            || is_allowed_vec_method(cx, parent))
                    {
                        ControlFlow::Continue(())
                    } else {
                        ControlFlow::Break(())
                    }
                })
                .is_continue();

                let span = expr.span.ctxt().outer_expn_data().call_site;
                if only_slice_uses {
                    self.check_vec_macro(cx, &vec_args, span, expr.hir_id, SuggestedType::Array);
                } else {
                    self.span_to_lint_map.insert(span, None);
                }
            }
            // if the local pattern has a specified type, do not lint.
            else if let Some(_) = higher::VecArgs::hir(cx, expr)
                && let Node::Local(local) = cx.tcx.parent_hir_node(expr.hir_id)
                && local.ty.is_some()
            {
                let span = expr.span.ctxt().outer_expn_data().call_site;
                self.span_to_lint_map.insert(span, None);
            }
            // search for `for _ in vec![...]`
            else if let Some(parent) = get_parent_expr(cx, expr)
                && parent.span.is_desugaring(DesugaringKind::ForLoop)
                && self.msrv.meets(msrvs::ARRAY_INTO_ITERATOR)
            {
                // report the error around the `vec!` not inside `<std macros>:`
                let span = expr.span.ctxt().outer_expn_data().call_site;
                self.check_vec_macro(cx, &vec_args, span, expr.hir_id, SuggestedType::Array);
            }
            // search for `&vec![_]` or `vec![_]` expressions where the adjusted type is `&[_]`
            else {
                let (suggest_slice, span) = if let ExprKind::AddrOf(BorrowKind::Ref, mutability, _) = expr.kind {
                    // `expr` is `&vec![_]`, so suggest `&[_]` (or `&mut[_]` resp.)
                    (SuggestedType::SliceRef(mutability), expr.span)
                } else {
                    // `expr` is the `vec![_]` expansion, so suggest `[_]`
                    // and also use the span of the actual `vec![_]` expression
                    (SuggestedType::Array, expr.span.ctxt().outer_expn_data().call_site)
                };

                if adjusts_to_slice(cx, expr) {
                    self.check_vec_macro(cx, &vec_args, span, expr.hir_id, suggest_slice);
                } else {
                    self.span_to_lint_map.insert(span, None);
                }
            }
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        for (span, lint_opt) in &self.span_to_lint_map {
            if let Some((hir_id, suggest_slice, snippet, applicability)) = lint_opt {
                let help_msg = format!(
                    "you can use {} directly",
                    match suggest_slice {
                        SuggestedType::SliceRef(_) => "a slice",
                        SuggestedType::Array => "an array",
                    }
                );
                span_lint_hir_and_then(cx, USELESS_VEC, *hir_id, *span, "useless use of `vec!`", |diag| {
                    diag.span_suggestion(*span, help_msg, snippet, *applicability);
                });
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

#[derive(Copy, Clone)]
pub(crate) enum SuggestedType {
    /// Suggest using a slice `&[..]` / `&mut [..]`
    SliceRef(Mutability),
    /// Suggest using an array: `[..]`
    Array,
}

impl UselessVec {
    fn check_vec_macro<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        vec_args: &higher::VecArgs<'tcx>,
        span: Span,
        hir_id: HirId,
        suggest_slice: SuggestedType,
    ) {
        if span.from_expansion() {
            return;
        }

        let mut applicability = Applicability::MachineApplicable;

        let snippet = match *vec_args {
            higher::VecArgs::Repeat(elem, len) => {
                if let Some(Constant::Int(len_constant)) = constant(cx, cx.typeck_results(), len) {
                    // vec![ty; N] works when ty is Clone, [ty; N] requires it to be Copy also
                    if !is_copy(cx, cx.typeck_results().expr_ty(elem)) {
                        return;
                    }

                    #[expect(clippy::cast_possible_truncation)]
                    if len_constant as u64 * size_of(cx, elem) > self.too_large_for_stack {
                        return;
                    }

                    let elem = snippet_with_applicability(cx, elem.span, "elem", &mut applicability);
                    let len = snippet_with_applicability(cx, len.span, "len", &mut applicability);

                    match suggest_slice {
                        SuggestedType::SliceRef(Mutability::Mut) => format!("&mut [{elem}; {len}]"),
                        SuggestedType::SliceRef(Mutability::Not) => format!("&[{elem}; {len}]"),
                        SuggestedType::Array => format!("[{elem}; {len}]"),
                    }
                } else {
                    return;
                }
            },
            higher::VecArgs::Vec(args) => {
                if let Some(last) = args.iter().last() {
                    if args.len() as u64 * size_of(cx, last) > self.too_large_for_stack {
                        return;
                    }
                    let span = args[0].span.source_callsite().to(last.span.source_callsite());
                    let args = snippet_with_applicability(cx, span, "..", &mut applicability);

                    match suggest_slice {
                        SuggestedType::SliceRef(Mutability::Mut) => {
                            format!("&mut [{args}]")
                        },
                        SuggestedType::SliceRef(Mutability::Not) => {
                            format!("&[{args}]")
                        },
                        SuggestedType::Array => {
                            format!("[{args}]")
                        },
                    }
                } else {
                    match suggest_slice {
                        SuggestedType::SliceRef(Mutability::Mut) => "&mut []".to_owned(),
                        SuggestedType::SliceRef(Mutability::Not) => "&[]".to_owned(),
                        SuggestedType::Array => "[]".to_owned(),
                    }
                }
            },
        };

        self.span_to_lint_map
            .entry(span)
            .or_insert(Some((hir_id, suggest_slice, snippet, applicability)));
    }
}

fn size_of(cx: &LateContext<'_>, expr: &Expr<'_>) -> u64 {
    let ty = cx.typeck_results().expr_ty_adjusted(expr);
    cx.layout_of(ty).map_or(0, |l| l.size.bytes())
}
