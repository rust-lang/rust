use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::higher::VecArgs;
use clippy_utils::source::{snippet_opt, snippet_with_applicability};
use clippy_utils::ty::get_type_diagnostic_name;
use clippy_utils::usage::{local_used_after_expr, local_used_in};
use clippy_utils::{
    get_path_from_caller_to_method_type, is_adjusted, is_no_std_crate, path_to_local, path_to_local_id,
};
use rustc_abi::ExternAbi;
use rustc_attr_data_structures::{AttributeKind, find_attr};
use rustc_errors::Applicability;
use rustc_hir::{BindingMode, Expr, ExprKind, FnRetTy, GenericArgs, Param, PatKind, QPath, Safety, TyKind};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{
    self, Binder, ClosureKind, FnSig, GenericArg, GenericArgKind, List, Region, Ty, TypeVisitableExt, TypeckResults,
};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::sym;
use rustc_trait_selection::error_reporting::InferCtxtErrorExt as _;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for closures which just call another function where
    /// the function can be called directly. `unsafe` functions, calls where types
    /// get adjusted or where the callee is marked `#[track_caller]` are ignored.
    ///
    /// ### Why is this bad?
    /// Needlessly creating a closure adds code for no benefit
    /// and gives the optimizer more work.
    ///
    /// ### Known problems
    /// If creating the closure inside the closure has a side-
    /// effect then moving the closure creation out will change when that side-
    /// effect runs.
    /// See [#1439](https://github.com/rust-lang/rust-clippy/issues/1439) for more details.
    ///
    /// ### Example
    /// ```rust,ignore
    /// xs.map(|x| foo(x))
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// // where `foo(_)` is a plain function that takes the exact argument type of `x`.
    /// xs.map(foo)
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub REDUNDANT_CLOSURE,
    style,
    "redundant closures, i.e., `|a| foo(a)` (which can be written as just `foo`)"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for closures which only invoke a method on the closure
    /// argument and can be replaced by referencing the method directly.
    ///
    /// ### Why is this bad?
    /// It's unnecessary to create the closure.
    ///
    /// ### Example
    /// ```rust,ignore
    /// Some('a').map(|s| s.to_uppercase());
    /// ```
    /// may be rewritten as
    /// ```rust,ignore
    /// Some('a').map(char::to_uppercase);
    /// ```
    #[clippy::version = "1.35.0"]
    pub REDUNDANT_CLOSURE_FOR_METHOD_CALLS,
    pedantic,
    "redundant closures for method calls"
}

declare_lint_pass!(EtaReduction => [REDUNDANT_CLOSURE, REDUNDANT_CLOSURE_FOR_METHOD_CALLS]);

impl<'tcx> LateLintPass<'tcx> for EtaReduction {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if let ExprKind::MethodCall(_method, receiver, args, _) = expr.kind {
            for arg in args {
                check_closure(cx, Some(receiver), arg);
            }
        }
        if let ExprKind::Call(func, args) = expr.kind {
            check_closure(cx, None, func);
            for arg in args {
                check_closure(cx, None, arg);
            }
        }
    }
}

#[allow(clippy::too_many_lines)]
fn check_closure<'tcx>(cx: &LateContext<'tcx>, outer_receiver: Option<&Expr<'tcx>>, expr: &Expr<'tcx>) {
    let body = if let ExprKind::Closure(c) = expr.kind
        && c.fn_decl.inputs.iter().all(|ty| matches!(ty.kind, TyKind::Infer(())))
        && matches!(c.fn_decl.output, FnRetTy::DefaultReturn(_))
        && !expr.span.from_expansion()
    {
        cx.tcx.hir_body(c.body)
    } else {
        return;
    };

    if body.value.span.from_expansion() {
        if body.params.is_empty()
            && let Some(VecArgs::Vec(&[])) = VecArgs::hir(cx, body.value)
        {
            let vec_crate = if is_no_std_crate(cx) { "alloc" } else { "std" };
            // replace `|| vec![]` with `Vec::new`
            span_lint_and_sugg(
                cx,
                REDUNDANT_CLOSURE,
                expr.span,
                "redundant closure",
                "replace the closure with `Vec::new`",
                format!("{vec_crate}::vec::Vec::new"),
                Applicability::MachineApplicable,
            );
        }
        // skip `foo(|| macro!())`
        return;
    }

    if is_adjusted(cx, body.value) {
        return;
    }

    let typeck = cx.typeck_results();
    let closure = if let ty::Closure(_, closure_subs) = typeck.expr_ty(expr).kind() {
        closure_subs.as_closure()
    } else {
        return;
    };
    let closure_sig = cx.tcx.signature_unclosure(closure.sig(), Safety::Safe).skip_binder();
    match body.value.kind {
        ExprKind::Call(callee, args)
            if matches!(
                callee.kind,
                ExprKind::Path(QPath::Resolved(..) | QPath::TypeRelative(..))
            ) =>
        {
            let callee_ty_raw = typeck.expr_ty(callee);
            let callee_ty = callee_ty_raw.peel_refs();
            if matches!(get_type_diagnostic_name(cx, callee_ty), Some(sym::Arc | sym::Rc))
                || !check_inputs(typeck, body.params, None, args)
            {
                return;
            }
            let callee_ty_adjusted = typeck
                .expr_adjustments(callee)
                .last()
                .map_or(callee_ty, |a| a.target.peel_refs());

            let sig = match callee_ty_adjusted.kind() {
                ty::FnDef(def, _) => {
                    // Rewriting `x(|| f())` to `x(f)` where f is marked `#[track_caller]` moves the `Location`
                    if find_attr!(cx.tcx.get_all_attrs(*def), AttributeKind::TrackCaller(..)) {
                        return;
                    }

                    cx.tcx.fn_sig(def).skip_binder().skip_binder()
                },
                ty::FnPtr(sig_tys, hdr) => sig_tys.with(*hdr).skip_binder(),
                ty::Closure(_, subs) => cx
                    .tcx
                    .signature_unclosure(subs.as_closure().sig(), Safety::Safe)
                    .skip_binder(),
                _ => {
                    if typeck.type_dependent_def_id(body.value.hir_id).is_some()
                        && let subs = typeck.node_args(body.value.hir_id)
                        && let output = typeck.expr_ty(body.value)
                        && let ty::Tuple(tys) = *subs.type_at(1).kind()
                    {
                        cx.tcx.mk_fn_sig(tys, output, false, Safety::Safe, ExternAbi::Rust)
                    } else {
                        return;
                    }
                },
            };
            if let Some(outer) = outer_receiver
                && ty_has_static(sig.output())
                && let generic_args = typeck.node_args(outer.hir_id)
                // HACK: Given a closure in `T.method(|| f())`, where `fn f() -> U where U: 'static`, `T.method(f)`
                // will succeed iff `T: 'static`. But the region of `T` is always erased by `typeck.expr_ty()` when
                // T is a generic type. For example, return type of `Option<String>::as_deref()` is a generic.
                // So we have a hack like this.
                && !generic_args.is_empty()
            {
                return;
            }
            if check_sig(closure_sig, sig)
                && let generic_args = typeck.node_args(callee.hir_id)
                // Given some trait fn `fn f() -> ()` and some type `T: Trait`, `T::f` is not
                // `'static` unless `T: 'static`. The cast `T::f as fn()` will, however, result
                // in a type which is `'static`.
                // For now ignore all callee types which reference a type parameter.
                && !generic_args.types().any(|t| matches!(t.kind(), ty::Param(_)))
            {
                span_lint_and_then(cx, REDUNDANT_CLOSURE, expr.span, "redundant closure", |diag| {
                    if let Some(mut snippet) = snippet_opt(cx, callee.span) {
                        if path_to_local(callee).is_some_and(|l| {
                            // FIXME: Do we really need this `local_used_in` check?
                            // Isn't it checking something like... `callee(callee)`?
                            // If somehow this check is needed, add some test for it,
                            // 'cuz currently nothing changes after deleting this check.
                            local_used_in(cx, l, args) || local_used_after_expr(cx, l, expr)
                        }) {
                            match cx
                                .tcx
                                .infer_ctxt()
                                .build(cx.typing_mode())
                                .err_ctxt()
                                .type_implements_fn_trait(
                                    cx.param_env,
                                    Binder::bind_with_vars(callee_ty_adjusted, List::empty()),
                                    ty::PredicatePolarity::Positive,
                                ) {
                                // Mutable closure is used after current expr; we cannot consume it.
                                Ok((ClosureKind::FnMut, _)) => snippet = format!("&mut {snippet}"),
                                Ok((ClosureKind::Fn, _)) if !callee_ty_raw.is_ref() => {
                                    snippet = format!("&{snippet}");
                                },
                                _ => (),
                            }
                        }
                        diag.span_suggestion(
                            expr.span,
                            "replace the closure with the function itself",
                            snippet,
                            Applicability::MachineApplicable,
                        );
                    }
                });
            }
        },
        ExprKind::MethodCall(path, self_, args, _) if check_inputs(typeck, body.params, Some(self_), args) => {
            if let Some(method_def_id) = typeck.type_dependent_def_id(body.value.hir_id)
                && !find_attr!(cx.tcx.get_all_attrs(method_def_id), AttributeKind::TrackCaller(..))
                && check_sig(closure_sig, cx.tcx.fn_sig(method_def_id).skip_binder().skip_binder())
            {
                let mut app = Applicability::MachineApplicable;
                let generic_args = match path.args.and_then(GenericArgs::span_ext) {
                    Some(span) => format!("::{}", snippet_with_applicability(cx, span, "<..>", &mut app)),
                    None => String::new(),
                };
                span_lint_and_then(
                    cx,
                    REDUNDANT_CLOSURE_FOR_METHOD_CALLS,
                    expr.span,
                    "redundant closure",
                    |diag| {
                        let args = typeck.node_args(body.value.hir_id);
                        let caller = self_.hir_id.owner.def_id;
                        let type_name = get_path_from_caller_to_method_type(cx.tcx, caller, method_def_id, args);
                        diag.span_suggestion(
                            expr.span,
                            "replace the closure with the method itself",
                            format!("{}::{}{}", type_name, path.ident.name, generic_args),
                            app,
                        );
                    },
                );
            }
        },
        _ => (),
    }
}

fn check_inputs(
    typeck: &TypeckResults<'_>,
    params: &[Param<'_>],
    self_arg: Option<&Expr<'_>>,
    args: &[Expr<'_>],
) -> bool {
    params.len() == self_arg.map_or(0, |_| 1) + args.len()
        && params.iter().zip(self_arg.into_iter().chain(args)).all(|(p, arg)| {
            matches!(
                p.pat.kind,
                PatKind::Binding(BindingMode::NONE, id, _, None)
                if path_to_local_id(arg, id)
            )
            // Only allow adjustments which change regions (i.e. re-borrowing).
            && typeck
                .expr_adjustments(arg)
                .last()
                .is_none_or(|a| a.target == typeck.expr_ty(arg))
        })
}

fn check_sig<'tcx>(closure_sig: FnSig<'tcx>, call_sig: FnSig<'tcx>) -> bool {
    call_sig.safety.is_safe() && !has_late_bound_to_non_late_bound_regions(closure_sig, call_sig)
}

/// This walks through both signatures and checks for any time a late-bound region is expected by an
/// `impl Fn` type, but the target signature does not have a late-bound region in the same position.
///
/// This is needed because rustc is unable to late bind early-bound regions in a function signature.
fn has_late_bound_to_non_late_bound_regions(from_sig: FnSig<'_>, to_sig: FnSig<'_>) -> bool {
    fn check_region(from_region: Region<'_>, to_region: Region<'_>) -> bool {
        from_region.is_bound() && !to_region.is_bound()
    }

    fn check_subs(from_subs: &[GenericArg<'_>], to_subs: &[GenericArg<'_>]) -> bool {
        if from_subs.len() != to_subs.len() {
            return true;
        }
        for (from_arg, to_arg) in to_subs.iter().zip(from_subs) {
            match (from_arg.kind(), to_arg.kind()) {
                (GenericArgKind::Lifetime(from_region), GenericArgKind::Lifetime(to_region)) => {
                    if check_region(from_region, to_region) {
                        return true;
                    }
                },
                (GenericArgKind::Type(from_ty), GenericArgKind::Type(to_ty)) => {
                    if check_ty(from_ty, to_ty) {
                        return true;
                    }
                },
                (GenericArgKind::Const(_), GenericArgKind::Const(_)) => (),
                _ => return true,
            }
        }
        false
    }

    fn check_ty(from_ty: Ty<'_>, to_ty: Ty<'_>) -> bool {
        match (from_ty.kind(), to_ty.kind()) {
            (&ty::Adt(_, from_subs), &ty::Adt(_, to_subs)) => check_subs(from_subs, to_subs),
            (&ty::Array(from_ty, _), &ty::Array(to_ty, _)) | (&ty::Slice(from_ty), &ty::Slice(to_ty)) => {
                check_ty(from_ty, to_ty)
            },
            (&ty::Ref(from_region, from_ty, _), &ty::Ref(to_region, to_ty, _)) => {
                check_region(from_region, to_region) || check_ty(from_ty, to_ty)
            },
            (&ty::Tuple(from_tys), &ty::Tuple(to_tys)) => {
                from_tys.len() != to_tys.len()
                    || from_tys
                        .iter()
                        .zip(to_tys)
                        .any(|(from_ty, to_ty)| check_ty(from_ty, to_ty))
            },
            _ => from_ty.has_bound_regions(),
        }
    }

    assert!(from_sig.inputs_and_output.len() == to_sig.inputs_and_output.len());
    from_sig
        .inputs_and_output
        .iter()
        .zip(to_sig.inputs_and_output)
        .any(|(from_ty, to_ty)| check_ty(from_ty, to_ty))
}

fn ty_has_static(ty: Ty<'_>) -> bool {
    ty.walk()
        .any(|arg| matches!(arg.kind(), GenericArgKind::Lifetime(re) if re.is_static()))
}
