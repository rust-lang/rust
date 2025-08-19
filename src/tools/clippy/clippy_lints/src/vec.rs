use std::collections::BTreeMap;
use std::ops::ControlFlow;

use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::ty::is_copy;
use clippy_utils::visitors::for_each_local_use_after_expr;
use clippy_utils::{get_parent_expr, higher, is_in_test, is_trait_method, span_contains_comment, sym};
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, HirId, LetStmt, Mutability, Node, Pat, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_middle::ty::layout::LayoutOf;
use rustc_session::impl_lint_pass;
use rustc_span::{DesugaringKind, Span};

pub struct UselessVec {
    too_large_for_stack: u64,
    msrv: Msrv,
    span_to_lint_map: BTreeMap<Span, Option<(HirId, SuggestedType, String, Applicability)>>,
    allow_in_test: bool,
}
impl UselessVec {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            too_large_for_stack: conf.too_large_for_stack,
            msrv: conf.msrv,
            span_to_lint_map: BTreeMap::new(),
            allow_in_test: conf.allow_useless_vec_in_tests,
        }
    }
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

impl<'tcx> LateLintPass<'tcx> for UselessVec {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let Some(vec_args) = higher::VecArgs::hir(cx, expr.peel_borrows()) else {
            return;
        };
        if self.allow_in_test && is_in_test(cx.tcx, expr.hir_id) {
            return;
        }
        // the parent callsite of this `vec!` expression, or span to the borrowed one such as `&vec!`
        let callsite = expr.span.parent_callsite().unwrap_or(expr.span);

        match cx.tcx.parent_hir_node(expr.hir_id) {
            // search for `let foo = vec![_]` expressions where all uses of `foo`
            // adjust to slices or call a method that exist on slices (e.g. len)
            Node::LetStmt(LetStmt {
                ty: None,
                pat:
                    Pat {
                        kind: PatKind::Binding(_, id, ..),
                        ..
                    },
                ..
            }) => {
                let only_slice_uses = for_each_local_use_after_expr(cx, *id, expr.hir_id, |expr| {
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

                if only_slice_uses {
                    self.check_vec_macro(cx, &vec_args, callsite, expr.hir_id, SuggestedType::Array);
                } else {
                    self.span_to_lint_map.insert(callsite, None);
                }
            },
            // if the local pattern has a specified type, do not lint.
            Node::LetStmt(LetStmt { ty: Some(_), .. }) if higher::VecArgs::hir(cx, expr).is_some() => {
                self.span_to_lint_map.insert(callsite, None);
            },
            // search for `for _ in vec![...]`
            Node::Expr(Expr { span, .. })
                if span.is_desugaring(DesugaringKind::ForLoop) && self.msrv.meets(cx, msrvs::ARRAY_INTO_ITERATOR) =>
            {
                let suggest_slice = suggest_type(expr);
                self.check_vec_macro(cx, &vec_args, callsite, expr.hir_id, suggest_slice);
            },
            // search for `&vec![_]` or `vec![_]` expressions where the adjusted type is `&[_]`
            _ => {
                let suggest_slice = suggest_type(expr);

                if adjusts_to_slice(cx, expr) {
                    self.check_vec_macro(cx, &vec_args, callsite, expr.hir_id, suggest_slice);
                } else {
                    self.span_to_lint_map.insert(callsite, None);
                }
            },
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        for (span, lint_opt) in &self.span_to_lint_map {
            if let Some((hir_id, suggest_slice, snippet, applicability)) = lint_opt {
                let help_msg = format!("you can use {} directly", suggest_slice.desc());
                span_lint_hir_and_then(cx, USELESS_VEC, *hir_id, *span, "useless use of `vec!`", |diag| {
                    // If the `vec!` macro contains comment, better not make the suggestion machine
                    // applicable as it would remove them.
                    let applicability = if *applicability != Applicability::Unspecified
                        && let source_map = cx.tcx.sess.source_map()
                        && span_contains_comment(source_map, *span)
                    {
                        Applicability::Unspecified
                    } else {
                        *applicability
                    };
                    diag.span_suggestion(*span, help_msg, snippet, applicability);
                });
            }
        }
    }
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

        let snippet = match *vec_args {
            higher::VecArgs::Repeat(elem, len) => {
                if let Some(Constant::Int(len_constant)) = ConstEvalCtxt::new(cx).eval(len) {
                    // vec![ty; N] works when ty is Clone, [ty; N] requires it to be Copy also
                    if !is_copy(cx, cx.typeck_results().expr_ty(elem)) {
                        return;
                    }

                    #[expect(clippy::cast_possible_truncation)]
                    if len_constant as u64 * size_of(cx, elem) > self.too_large_for_stack {
                        return;
                    }

                    suggest_slice.snippet(cx, Some(elem.span), Some(len.span))
                } else {
                    return;
                }
            },
            higher::VecArgs::Vec(args) => {
                let args_span = if let Some(last) = args.iter().last() {
                    if args.len() as u64 * size_of(cx, last) > self.too_large_for_stack {
                        return;
                    }
                    Some(args[0].span.source_callsite().to(last.span.source_callsite()))
                } else {
                    None
                };
                suggest_slice.snippet(cx, args_span, None)
            },
        };

        self.span_to_lint_map.entry(span).or_insert(Some((
            hir_id,
            suggest_slice,
            snippet,
            Applicability::MachineApplicable,
        )));
    }
}

#[derive(Copy, Clone)]
pub(crate) enum SuggestedType {
    /// Suggest using a slice `&[..]` / `&mut [..]`
    SliceRef(Mutability),
    /// Suggest using an array: `[..]`
    Array,
}

impl SuggestedType {
    fn desc(self) -> &'static str {
        match self {
            Self::SliceRef(_) => "a slice",
            Self::Array => "an array",
        }
    }

    fn snippet(self, cx: &LateContext<'_>, args_span: Option<Span>, len_span: Option<Span>) -> String {
        let maybe_args = args_span
            .and_then(|sp| sp.get_source_text(cx))
            .map_or(String::new(), |x| x.to_owned());
        let maybe_len = len_span
            .and_then(|sp| sp.get_source_text(cx).map(|s| format!("; {s}")))
            .unwrap_or_default();

        match self {
            Self::SliceRef(Mutability::Mut) => format!("&mut [{maybe_args}{maybe_len}]"),
            Self::SliceRef(Mutability::Not) => format!("&[{maybe_args}{maybe_len}]"),
            Self::Array => format!("[{maybe_args}{maybe_len}]"),
        }
    }
}

fn size_of(cx: &LateContext<'_>, expr: &Expr<'_>) -> u64 {
    let ty = cx.typeck_results().expr_ty_adjusted(expr);
    cx.layout_of(ty).map_or(0, |l| l.size.bytes())
}

fn adjusts_to_slice(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    matches!(cx.typeck_results().expr_ty_adjusted(e).kind(), ty::Ref(_, ty, _) if ty.is_slice())
}

/// Checks if the given expression is a method call to a `Vec` method
/// that also exists on slices. If this returns true, it means that
/// this expression does not actually require a `Vec` and could just work with an array.
pub fn is_allowed_vec_method(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    if let ExprKind::MethodCall(path, _, [], _) = e.kind {
        matches!(path.ident.name, sym::as_ptr | sym::is_empty | sym::len)
    } else {
        is_trait_method(cx, e, sym::IntoIterator)
    }
}

fn suggest_type(expr: &Expr<'_>) -> SuggestedType {
    if let ExprKind::AddrOf(BorrowKind::Ref, mutability, _) = expr.kind {
        // `expr` is `&vec![_]`, so suggest `&[_]` (or `&mut[_]` resp.)
        SuggestedType::SliceRef(mutability)
    } else {
        // `expr` is the `vec![_]` expansion, so suggest `[_]`
        SuggestedType::Array
    }
}
