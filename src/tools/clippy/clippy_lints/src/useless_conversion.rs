use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::is_ty_alias;
use clippy_utils::source::{snippet, snippet_with_applicability, snippet_with_context};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{is_copy, is_type_diagnostic_item, same_type_and_consts};
use clippy_utils::{get_parent_expr, is_trait_method, match_def_path, path_to_local, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{BindingAnnotation, Expr, ExprKind, HirId, MatchSource, Node, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `Into`, `TryInto`, `From`, `TryFrom`, or `IntoIter` calls
    /// which uselessly convert to the same type.
    ///
    /// ### Why is this bad?
    /// Redundant code.
    ///
    /// ### Example
    /// ```rust
    /// // format!() returns a `String`
    /// let s: String = format!("hello").into();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let s: String = format!("hello");
    /// ```
    #[clippy::version = "1.45.0"]
    pub USELESS_CONVERSION,
    complexity,
    "calls to `Into`, `TryInto`, `From`, `TryFrom`, or `IntoIter` which perform useless conversions to the same type"
}

#[derive(Default)]
pub struct UselessConversion {
    try_desugar_arm: Vec<HirId>,
}

impl_lint_pass!(UselessConversion => [USELESS_CONVERSION]);

enum MethodOrFunction {
    Method,
    Function,
}

impl MethodOrFunction {
    /// Maps the argument position in `pos` to the parameter position.
    /// For methods, `self` is skipped.
    fn param_pos(self, pos: usize) -> usize {
        match self {
            MethodOrFunction::Method => pos + 1,
            MethodOrFunction::Function => pos,
        }
    }
}

/// Returns the span of the `IntoIterator` trait bound in the function pointed to by `fn_did`
fn into_iter_bound(cx: &LateContext<'_>, fn_did: DefId, into_iter_did: DefId, param_index: u32) -> Option<Span> {
    cx.tcx
        .predicates_of(fn_did)
        .predicates
        .iter()
        .find_map(|&(ref pred, span)| {
            if let ty::ClauseKind::Trait(tr) = pred.kind().skip_binder()
                && tr.def_id() == into_iter_did
                && tr.self_ty().is_param(param_index)
            {
                Some(span)
            } else {
                None
            }
        })
}

/// Extracts the receiver of a `.into_iter()` method call.
fn into_iter_call<'hir>(cx: &LateContext<'_>, expr: &'hir Expr<'hir>) -> Option<&'hir Expr<'hir>> {
    if let ExprKind::MethodCall(name, recv, _, _) = expr.kind
        && is_trait_method(cx, expr, sym::IntoIterator)
        && name.ident.name == sym::into_iter
    {
        Some(recv)
    } else {
        None
    }
}

/// Same as [`into_iter_call`], but tries to look for the innermost `.into_iter()` call, e.g.:
/// `foo.into_iter().into_iter()`
///  ^^^  we want this expression
fn into_iter_deep_call<'hir>(cx: &LateContext<'_>, mut expr: &'hir Expr<'hir>) -> (&'hir Expr<'hir>, usize) {
    let mut depth = 0;
    while let Some(recv) = into_iter_call(cx, expr) {
        expr = recv;
        depth += 1;
    }
    (expr, depth)
}

#[expect(clippy::too_many_lines)]
impl<'tcx> LateLintPass<'tcx> for UselessConversion {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if e.span.from_expansion() {
            return;
        }

        if Some(&e.hir_id) == self.try_desugar_arm.last() {
            return;
        }

        match e.kind {
            ExprKind::Match(_, arms, MatchSource::TryDesugar) => {
                let (ExprKind::Ret(Some(e)) | ExprKind::Break(_, Some(e))) = arms[0].body.kind else {
                     return
                };
                if let ExprKind::Call(_, [arg, ..]) = e.kind {
                    self.try_desugar_arm.push(arg.hir_id);
                }
            },

            ExprKind::MethodCall(name, recv, ..) => {
                if is_trait_method(cx, e, sym::Into) && name.ident.as_str() == "into" {
                    let a = cx.typeck_results().expr_ty(e);
                    let b = cx.typeck_results().expr_ty(recv);
                    if same_type_and_consts(a, b) {
                        let mut app = Applicability::MachineApplicable;
                        let sugg = snippet_with_context(cx, recv.span, e.span.ctxt(), "<expr>", &mut app).0;
                        span_lint_and_sugg(
                            cx,
                            USELESS_CONVERSION,
                            e.span,
                            &format!("useless conversion to the same type: `{b}`"),
                            "consider removing `.into()`",
                            sugg.into_owned(),
                            app,
                        );
                    }
                }
                if let Some(into_iter_recv) = into_iter_call(cx, e)
                    // Make sure that there is no parent expression, or if there is, make sure it's not a `.into_iter()` call.
                    // The reason for that is that we only want to lint once (the outermost call)
                    // in cases like `foo.into_iter().into_iter()`
                    && get_parent_expr(cx, e)
                        .and_then(|parent| into_iter_call(cx, parent))
                        .is_none()
                {
                    if let Some(parent) = get_parent_expr(cx, e) {
                        let parent_fn = match parent.kind {
                            ExprKind::Call(recv, args) if let ExprKind::Path(ref qpath) = recv.kind => {
                                cx.qpath_res(qpath, recv.hir_id).opt_def_id()
                                    .map(|did| (did, args, MethodOrFunction::Function))
                            }
                            ExprKind::MethodCall(.., args, _) => {
                                cx.typeck_results().type_dependent_def_id(parent.hir_id)
                                    .map(|did| (did, args, MethodOrFunction::Method))
                            }
                            _ => None,
                        };

                        if let Some((parent_fn_did, args, kind)) = parent_fn
                            && let Some(into_iter_did) = cx.tcx.get_diagnostic_item(sym::IntoIterator)
                            && let sig = cx.tcx.fn_sig(parent_fn_did).skip_binder().skip_binder()
                            && let Some(arg_pos) = args.iter().position(|x| x.hir_id == e.hir_id)
                            && let Some(&into_iter_param) = sig.inputs().get(kind.param_pos(arg_pos))
                            && let ty::Param(param) = into_iter_param.kind()
                            && let Some(span) = into_iter_bound(cx, parent_fn_did, into_iter_did, param.index)
                        {
                            // Get the "innermost" `.into_iter()` call, e.g. given this expression:
                            // `foo.into_iter().into_iter()`
                            //  ^^^
                            let (into_iter_recv, depth) = into_iter_deep_call(cx, into_iter_recv);

                            let plural = if depth == 0 { "" } else { "s" };
                            let mut applicability = Applicability::MachineApplicable;
                            let sugg = snippet_with_applicability(cx, into_iter_recv.span.source_callsite(), "<expr>", &mut applicability).into_owned();
                            span_lint_and_then(cx, USELESS_CONVERSION, e.span, "explicit call to `.into_iter()` in function argument accepting `IntoIterator`", |diag| {
                                diag.span_suggestion(
                                    e.span,
                                    format!("consider removing the `.into_iter()`{plural}"),
                                    sugg,
                                    applicability,
                                );
                                diag.span_note(span, "this parameter accepts any `IntoIterator`, so you don't need to call `.into_iter()`");
                            });

                            // Early return to avoid linting again with contradicting suggestions
                            return;
                        }
                    }

                    if let Some(id) = path_to_local(recv) &&
                       let Node::Pat(pat) = cx.tcx.hir().get(id) &&
                       let PatKind::Binding(ann, ..) = pat.kind &&
                       ann != BindingAnnotation::MUT
                    {
                        // Do not remove .into_iter() applied to a non-mutable local variable used in
                        // a larger expression context as it would differ in mutability.
                        return;
                    }

                    let a = cx.typeck_results().expr_ty(e);
                    let b = cx.typeck_results().expr_ty(recv);

                    // If the types are identical then .into_iter() can be removed, unless the type
                    // implements Copy, in which case .into_iter() returns a copy of the receiver and
                    // cannot be safely omitted.
                    if same_type_and_consts(a, b) && !is_copy(cx, b) {
                        let sugg = snippet(cx, recv.span, "<expr>").into_owned();
                        span_lint_and_sugg(
                            cx,
                            USELESS_CONVERSION,
                            e.span,
                            &format!("useless conversion to the same type: `{b}`"),
                            "consider removing `.into_iter()`",
                            sugg,
                            Applicability::MachineApplicable, // snippet
                        );
                    }
                }
                if_chain! {
                    if is_trait_method(cx, e, sym::TryInto) && name.ident.name == sym::try_into;
                    let a = cx.typeck_results().expr_ty(e);
                    let b = cx.typeck_results().expr_ty(recv);
                    if is_type_diagnostic_item(cx, a, sym::Result);
                    if let ty::Adt(_, substs) = a.kind();
                    if let Some(a_type) = substs.types().next();
                    if same_type_and_consts(a_type, b);

                    then {
                        span_lint_and_help(
                            cx,
                            USELESS_CONVERSION,
                            e.span,
                            &format!("useless conversion to the same type: `{b}`"),
                            None,
                            "consider removing `.try_into()`",
                        );
                    }
                }
            },

            ExprKind::Call(path, [arg]) => {
                if_chain! {
                    if let ExprKind::Path(ref qpath) = path.kind;
                    if let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id();
                    if !is_ty_alias(qpath);
                    then {
                        let a = cx.typeck_results().expr_ty(e);
                        let b = cx.typeck_results().expr_ty(arg);
                        if_chain! {
                            if match_def_path(cx, def_id, &paths::TRY_FROM);
                            if is_type_diagnostic_item(cx, a, sym::Result);
                            if let ty::Adt(_, substs) = a.kind();
                            if let Some(a_type) = substs.types().next();
                            if same_type_and_consts(a_type, b);

                            then {
                                let hint = format!("consider removing `{}()`", snippet(cx, path.span, "TryFrom::try_from"));
                                span_lint_and_help(
                                    cx,
                                    USELESS_CONVERSION,
                                    e.span,
                                    &format!("useless conversion to the same type: `{b}`"),
                                    None,
                                    &hint,
                                );
                            }
                        }

                        if_chain! {
                            if cx.tcx.is_diagnostic_item(sym::from_fn, def_id);
                            if same_type_and_consts(a, b);

                            then {
                                let mut app = Applicability::MachineApplicable;
                                let sugg = Sugg::hir_with_context(cx, arg, e.span.ctxt(), "<expr>", &mut app).maybe_par();
                                let sugg_msg =
                                    format!("consider removing `{}()`", snippet(cx, path.span, "From::from"));
                                span_lint_and_sugg(
                                    cx,
                                    USELESS_CONVERSION,
                                    e.span,
                                    &format!("useless conversion to the same type: `{b}`"),
                                    &sugg_msg,
                                    sugg.to_string(),
                                    app,
                                );
                            }
                        }
                    }
                }
            },

            _ => {},
        }
    }

    fn check_expr_post(&mut self, _: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if Some(&e.hir_id) == self.try_desugar_arm.last() {
            self.try_desugar_arm.pop();
        }
    }
}
