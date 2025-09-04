use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::mem;
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
    /// Maps from a `vec![]` source callsite invocation span to the "state" (i.e., whether we can
    /// emit a warning there or not).
    ///
    /// The purpose of this is to buffer lints up until `check_crate_post` so that we can cancel a
    /// lint while visiting, because a `vec![]` invocation span can appear multiple times when
    /// it is passed as a macro argument, once in a context that doesn't require a `Vec<_>` and
    /// another time that does. Consider:
    /// ```
    /// macro_rules! m {
    ///     ($v:expr) => {
    ///         let a = $v;
    ///         $v.push(3);
    ///     }
    /// }
    /// m!(vec![1, 2]);
    /// ```
    /// The macro invocation expands to two `vec![1, 2]` invocations. If we eagerly suggest changing
    /// the first `vec![1, 2]` (which is shared with the other expn) to an array which indeed would
    /// work, we get a false positive warning on the `$v.push(3)` which really requires `$v` to
    /// be a vector.
    span_to_state: BTreeMap<Span, VecState>,
    allow_in_test: bool,
}

impl UselessVec {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            too_large_for_stack: conf.too_large_for_stack,
            msrv: conf.msrv,
            span_to_state: BTreeMap::new(),
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

/// The "state" of a `vec![]` invocation, indicating whether it can or cannot be changed.
enum VecState {
    Change {
        suggest_ty: SuggestedType,
        vec_snippet: String,
        expr_hir_id: HirId,
    },
    NoChange,
}

enum VecToArray {
    /// Expression does not need to be a `Vec<_>` and its type can be changed to an array (or
    /// slice).
    Possible,
    /// Expression must be a `Vec<_>`. Type cannot change.
    Impossible,
}

impl UselessVec {
    /// Checks if the surrounding environment requires this expression to actually be of type
    /// `Vec<_>`, or if it can be changed to `&[]`/`[]` without causing type errors.
    fn expr_usage_requires_vec(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) -> VecToArray {
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
                    VecToArray::Possible
                } else {
                    VecToArray::Impossible
                }
            },
            // if the local pattern has a specified type, do not lint.
            Node::LetStmt(LetStmt { ty: Some(_), .. }) if higher::VecArgs::hir(cx, expr).is_some() => {
                VecToArray::Impossible
            },
            // search for `for _ in vec![...]`
            Node::Expr(Expr { span, .. })
                if span.is_desugaring(DesugaringKind::ForLoop) && self.msrv.meets(cx, msrvs::ARRAY_INTO_ITERATOR) =>
            {
                VecToArray::Possible
            },
            // search for `&vec![_]` or `vec![_]` expressions where the adjusted type is `&[_]`
            _ => {
                if adjusts_to_slice(cx, expr) {
                    VecToArray::Possible
                } else {
                    VecToArray::Impossible
                }
            },
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for UselessVec {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let Some(vec_args) = higher::VecArgs::hir(cx, expr.peel_borrows())
            // The `vec![]` or `&vec![]` invocation span.
            && let vec_span = expr.span.parent_callsite().unwrap_or(expr.span)
            && !vec_span.from_expansion()
        {
            if self.allow_in_test && is_in_test(cx.tcx, expr.hir_id) {
                return;
            }

            match self.expr_usage_requires_vec(cx, expr) {
                VecToArray::Possible => {
                    let suggest_ty = suggest_type(expr);

                    // Size and `Copy` checks don't depend on the enclosing usage of the expression
                    // and don't need to be inserted into the state map.
                    let vec_snippet = match vec_args {
                        higher::VecArgs::Repeat(expr, len) => {
                            if is_copy(cx, cx.typeck_results().expr_ty(expr))
                                && let Some(Constant::Int(length)) = ConstEvalCtxt::new(cx).eval(len)
                                && let Ok(length) = u64::try_from(length)
                                && size_of(cx, expr)
                                    .checked_mul(length)
                                    .is_some_and(|size| size <= self.too_large_for_stack)
                            {
                                suggest_ty.snippet(
                                    cx,
                                    Some(expr.span.source_callsite()),
                                    Some(len.span.source_callsite()),
                                )
                            } else {
                                return;
                            }
                        },
                        higher::VecArgs::Vec(args) => {
                            if let Ok(length) = u64::try_from(args.len())
                                && size_of(cx, expr)
                                    .checked_mul(length)
                                    .is_some_and(|size| size <= self.too_large_for_stack)
                            {
                                suggest_ty.snippet(
                                    cx,
                                    args.first().zip(args.last()).map(|(first, last)| {
                                        first.span.source_callsite().to(last.span.source_callsite())
                                    }),
                                    None,
                                )
                            } else {
                                return;
                            }
                        },
                    };

                    if let Entry::Vacant(entry) = self.span_to_state.entry(vec_span) {
                        entry.insert(VecState::Change {
                            suggest_ty,
                            vec_snippet,
                            expr_hir_id: expr.hir_id,
                        });
                    }
                },
                VecToArray::Impossible => {
                    self.span_to_state.insert(vec_span, VecState::NoChange);
                },
            }
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        for (span, state) in mem::take(&mut self.span_to_state) {
            if let VecState::Change {
                suggest_ty,
                vec_snippet,
                expr_hir_id,
            } = state
            {
                span_lint_hir_and_then(cx, USELESS_VEC, expr_hir_id, span, "useless use of `vec!`", |diag| {
                    let help_msg = format!("you can use {} directly", suggest_ty.desc());
                    // If the `vec!` macro contains comment, better not make the suggestion machine applicable as it
                    // would remove them.
                    let applicability = if span_contains_comment(cx.tcx.sess.source_map(), span) {
                        Applicability::Unspecified
                    } else {
                        Applicability::MachineApplicable
                    };
                    diag.span_suggestion(span, help_msg, vec_snippet, applicability);
                });
            }
        }
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
        // Invariant of the lint as implemented: all spans are from the root context (and as a result,
        // always trivially crate-local).
        assert!(args_span.is_none_or(|s| !s.from_expansion()));
        assert!(len_span.is_none_or(|s| !s.from_expansion()));

        let maybe_args = args_span
            .map(|sp| sp.get_source_text(cx).expect("spans are always crate-local"))
            .map_or(String::new(), |x| x.to_owned());
        let maybe_len = len_span
            .map(|sp| sp.get_source_text(cx).expect("spans are always crate-local"))
            .map(|st| format!("; {st}"))
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
