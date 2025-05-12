use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::{snippet, snippet_with_context};
use clippy_utils::sugg::{DiagExt as _, Sugg};
use clippy_utils::ty::{get_type_diagnostic_name, is_copy, is_type_diagnostic_item, same_type_and_consts};
use clippy_utils::{
    get_parent_expr, is_inherent_method_call, is_trait_item, is_trait_method, is_ty_alias, path_to_local,
};
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{BindingMode, Expr, ExprKind, HirId, MatchSource, Node, PatKind};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::Obligation;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::adjustment::{Adjust, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::{self, EarlyBinder, GenericArg, GenericArgsRef, Ty, TypeVisitableExt};
use rustc_session::impl_lint_pass;
use rustc_span::{Span, sym};
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `Into`, `TryInto`, `From`, `TryFrom`, or `IntoIter` calls
    /// which uselessly convert to the same type.
    ///
    /// ### Why is this bad?
    /// Redundant code.
    ///
    /// ### Example
    /// ```no_run
    /// // format!() returns a `String`
    /// let s: String = format!("hello").into();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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
    expn_depth: u32,
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

/// Returns the span of the `IntoIterator` trait bound in the function pointed to by `fn_did`,
/// iff all of the bounds also hold for the type of the `.into_iter()` receiver.
/// ```ignore
/// pub fn foo<I>(i: I)
/// where I: IntoIterator<Item=i32> + ExactSizeIterator
///                                   ^^^^^^^^^^^^^^^^^ this extra bound stops us from suggesting to remove `.into_iter()` ...
/// {
///     assert_eq!(i.len(), 3);
/// }
///
/// pub fn bar() {
///     foo([1, 2, 3].into_iter());
///                  ^^^^^^^^^^^^ ... here, because `[i32; 3]` is not `ExactSizeIterator`
/// }
/// ```
fn into_iter_bound<'tcx>(
    cx: &LateContext<'tcx>,
    fn_did: DefId,
    into_iter_did: DefId,
    into_iter_receiver: Ty<'tcx>,
    param_index: u32,
    node_args: GenericArgsRef<'tcx>,
) -> Option<Span> {
    let mut into_iter_span = None;

    for (pred, span) in cx.tcx.explicit_predicates_of(fn_did).predicates {
        if let ty::ClauseKind::Trait(tr) = pred.kind().skip_binder()
            && tr.self_ty().is_param(param_index)
        {
            if tr.def_id() == into_iter_did {
                into_iter_span = Some(*span);
            } else {
                let tr = cx.tcx.erase_regions(tr);
                if tr.has_escaping_bound_vars() {
                    return None;
                }

                // Substitute generics in the predicate and replace the IntoIterator type parameter with the
                // `.into_iter()` receiver to see if the bound also holds for that type.
                let args = cx.tcx.mk_args_from_iter(node_args.iter().enumerate().map(|(i, arg)| {
                    if i == param_index as usize {
                        GenericArg::from(into_iter_receiver)
                    } else {
                        arg
                    }
                }));

                let predicate = EarlyBinder::bind(tr).instantiate(cx.tcx, args);
                let obligation = Obligation::new(cx.tcx, ObligationCause::dummy(), cx.param_env, predicate);
                if !cx
                    .tcx
                    .infer_ctxt()
                    .build(cx.typing_mode())
                    .predicate_must_hold_modulo_regions(&obligation)
                {
                    return None;
                }
            }
        }
    }

    into_iter_span
}

/// Extracts the receiver of a `.into_iter()` method call.
fn into_iter_call<'hir>(cx: &LateContext<'_>, expr: &'hir Expr<'hir>) -> Option<&'hir Expr<'hir>> {
    if let ExprKind::MethodCall(name, recv, [], _) = expr.kind
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
            self.expn_depth += 1;
            return;
        }

        if Some(&e.hir_id) == self.try_desugar_arm.last() {
            return;
        }

        match e.kind {
            ExprKind::Match(_, arms, MatchSource::TryDesugar(_)) => {
                let (ExprKind::Ret(Some(e)) | ExprKind::Break(_, Some(e))) = arms[0].body.kind else {
                    return;
                };
                if let ExprKind::Call(_, [arg, ..]) = e.kind {
                    self.try_desugar_arm.push(arg.hir_id);
                }
            },

            ExprKind::MethodCall(name, recv, [], _) => {
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
                            format!("useless conversion to the same type: `{b}`"),
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
                            ExprKind::Call(recv, args)
                                if let ExprKind::Path(ref qpath) = recv.kind
                                    && let Some(did) = cx.qpath_res(qpath, recv.hir_id).opt_def_id()
                                    // make sure that the path indeed points to a fn-like item, so that
                                    // `fn_sig` does not ICE. (see #11065)
                                    && cx.tcx.def_kind(did).is_fn_like() =>
                            {
                                Some((
                                    did,
                                    args,
                                    cx.typeck_results().node_args(recv.hir_id),
                                    MethodOrFunction::Function,
                                ))
                            },
                            ExprKind::MethodCall(.., args, _) => {
                                cx.typeck_results().type_dependent_def_id(parent.hir_id).map(|did| {
                                    (
                                        did,
                                        args,
                                        cx.typeck_results().node_args(parent.hir_id),
                                        MethodOrFunction::Method,
                                    )
                                })
                            },
                            _ => None,
                        };

                        if let Some((parent_fn_did, args, node_args, kind)) = parent_fn
                            && let Some(into_iter_did) = cx.tcx.get_diagnostic_item(sym::IntoIterator)
                            && let sig = cx.tcx.fn_sig(parent_fn_did).skip_binder().skip_binder()
                            && let Some(arg_pos) = args.iter().position(|x| x.hir_id == e.hir_id)
                            && let Some(&into_iter_param) = sig.inputs().get(kind.param_pos(arg_pos))
                            && let ty::Param(param) = into_iter_param.kind()
                            && let Some(span) = into_iter_bound(
                                cx,
                                parent_fn_did,
                                into_iter_did,
                                cx.typeck_results().expr_ty(into_iter_recv),
                                param.index,
                                node_args,
                            )
                            && self.expn_depth == 0
                        {
                            // Get the "innermost" `.into_iter()` call, e.g. given this expression:
                            // `foo.into_iter().into_iter()`
                            //  ^^^
                            let (into_iter_recv, depth) = into_iter_deep_call(cx, into_iter_recv);

                            span_lint_and_then(
                                cx,
                                USELESS_CONVERSION,
                                e.span,
                                "explicit call to `.into_iter()` in function argument accepting `IntoIterator`",
                                |diag| {
                                    let receiver_span = into_iter_recv.span.source_callsite();
                                    let adjustments = adjustments(cx, into_iter_recv);
                                    let mut sugg = if adjustments.is_empty() {
                                        vec![]
                                    } else {
                                        vec![(receiver_span.shrink_to_lo(), adjustments)]
                                    };
                                    let plural = if depth == 0 { "" } else { "s" };
                                    sugg.push((e.span.with_lo(receiver_span.hi()), String::new()));
                                    diag.multipart_suggestion(
                                        format!("consider removing the `.into_iter()`{plural}"),
                                        sugg,
                                        Applicability::MachineApplicable,
                                    );
                                    diag.span_note(span, "this parameter accepts any `IntoIterator`, so you don't need to call `.into_iter()`");
                                },
                            );

                            // Early return to avoid linting again with contradicting suggestions
                            return;
                        }
                    }

                    if let Some(id) = path_to_local(recv)
                        && let Node::Pat(pat) = cx.tcx.hir_node(id)
                        && let PatKind::Binding(ann, ..) = pat.kind
                        && ann != BindingMode::MUT
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
                            format!("useless conversion to the same type: `{b}`"),
                            "consider removing `.into_iter()`",
                            sugg,
                            Applicability::MachineApplicable, // snippet
                        );
                    }
                }
                if is_trait_method(cx, e, sym::TryInto)
                    && name.ident.name == sym::try_into
                    && let a = cx.typeck_results().expr_ty(e)
                    && let b = cx.typeck_results().expr_ty(recv)
                    && is_type_diagnostic_item(cx, a, sym::Result)
                    && let ty::Adt(_, args) = a.kind()
                    && let Some(a_type) = args.types().next()
                    && same_type_and_consts(a_type, b)
                {
                    span_lint_and_help(
                        cx,
                        USELESS_CONVERSION,
                        e.span,
                        format!("useless conversion to the same type: `{b}`"),
                        None,
                        "consider removing `.try_into()`",
                    );
                }
            },

            ExprKind::Call(path, [arg]) => {
                if let ExprKind::Path(ref qpath) = path.kind
                    && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
                    && !is_ty_alias(qpath)
                {
                    let a = cx.typeck_results().expr_ty(e);
                    let b = cx.typeck_results().expr_ty(arg);
                    if cx.tcx.is_diagnostic_item(sym::try_from_fn, def_id)
                        && is_type_diagnostic_item(cx, a, sym::Result)
                        && let ty::Adt(_, args) = a.kind()
                        && let Some(a_type) = args.types().next()
                        && same_type_and_consts(a_type, b)
                    {
                        let hint = format!("consider removing `{}()`", snippet(cx, path.span, "TryFrom::try_from"));
                        span_lint_and_help(
                            cx,
                            USELESS_CONVERSION,
                            e.span,
                            format!("useless conversion to the same type: `{b}`"),
                            None,
                            hint,
                        );
                    }

                    if cx.tcx.is_diagnostic_item(sym::from_fn, def_id) && same_type_and_consts(a, b) {
                        let mut app = Applicability::MachineApplicable;
                        let sugg = Sugg::hir_with_context(cx, arg, e.span.ctxt(), "<expr>", &mut app).maybe_paren();
                        let sugg_msg = format!("consider removing `{}()`", snippet(cx, path.span, "From::from"));
                        span_lint_and_sugg(
                            cx,
                            USELESS_CONVERSION,
                            e.span,
                            format!("useless conversion to the same type: `{b}`"),
                            sugg_msg,
                            sugg.to_string(),
                            app,
                        );
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
        if e.span.from_expansion() {
            self.expn_depth -= 1;
        }
    }
}

/// Check if `arg` is a `Into::into` or `From::from` applied to `receiver` to give `expr`, through a
/// higher-order mapping function.
pub fn check_function_application(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, arg: &Expr<'_>) {
    if has_eligible_receiver(cx, recv, expr)
        && (is_trait_item(cx, arg, sym::Into) || is_trait_item(cx, arg, sym::From))
        && let ty::FnDef(_, args) = cx.typeck_results().expr_ty(arg).kind()
        && let &[from_ty, to_ty] = args.into_type_list(cx.tcx).as_slice()
        && same_type_and_consts(from_ty, to_ty)
    {
        span_lint_and_then(
            cx,
            USELESS_CONVERSION,
            expr.span.with_lo(recv.span.hi()),
            format!("useless conversion to the same type: `{from_ty}`"),
            |diag| {
                diag.suggest_remove_item(
                    cx,
                    expr.span.with_lo(recv.span.hi()),
                    "consider removing",
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}

fn has_eligible_receiver(cx: &LateContext<'_>, recv: &Expr<'_>, expr: &Expr<'_>) -> bool {
    if is_inherent_method_call(cx, expr) {
        matches!(
            get_type_diagnostic_name(cx, cx.typeck_results().expr_ty(recv)),
            Some(sym::Option | sym::Result | sym::ControlFlow)
        )
    } else {
        is_trait_method(cx, expr, sym::Iterator)
    }
}

fn adjustments(cx: &LateContext<'_>, expr: &Expr<'_>) -> String {
    let mut prefix = String::new();
    for adj in cx.typeck_results().expr_adjustments(expr) {
        match adj.kind {
            Adjust::Deref(_) => prefix = format!("*{prefix}"),
            Adjust::Borrow(AutoBorrow::Ref(AutoBorrowMutability::Mut { .. })) => prefix = format!("&mut {prefix}"),
            Adjust::Borrow(AutoBorrow::Ref(AutoBorrowMutability::Not)) => prefix = format!("&{prefix}"),
            _ => {},
        }
    }
    prefix
}
