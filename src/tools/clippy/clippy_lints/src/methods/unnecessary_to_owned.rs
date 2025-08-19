use super::implicit_clone::is_clone_like;
use super::unnecessary_iter_cloned::{self, is_into_iter};
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{SpanRangeExt, snippet};
use clippy_utils::ty::{get_iterator_item_ty, implements_trait, is_copy, is_type_diagnostic_item, is_type_lang_item};
use clippy_utils::visitors::find_all_ret_expressions;
use clippy_utils::{
    fn_def_id, get_parent_expr, is_diag_item_method, is_diag_trait_item, is_expr_temporary_value, peel_middle_ty_refs,
    return_ty, sym,
};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{BorrowKind, Expr, ExprKind, ItemKind, LangItem, Node};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::LateContext;
use rustc_middle::mir::Mutability;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, OverloadedDeref};
use rustc_middle::ty::{
    self, ClauseKind, GenericArg, GenericArgKind, GenericArgsRef, ParamTy, ProjectionPredicate, TraitPredicate, Ty,
};
use rustc_span::Symbol;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_trait_selection::traits::{Obligation, ObligationCause};

use super::UNNECESSARY_TO_OWNED;

pub fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    method_name: Symbol,
    receiver: &'tcx Expr<'_>,
    args: &'tcx [Expr<'_>],
    msrv: Msrv,
) {
    if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && args.is_empty()
    {
        if is_cloned_or_copied(cx, method_name, method_def_id) {
            unnecessary_iter_cloned::check(cx, expr, method_name, receiver);
        } else if is_to_owned_like(cx, expr, method_name, method_def_id) {
            if check_split_call_arg(cx, expr, method_name, receiver) {
                return;
            }
            // At this point, we know the call is of a `to_owned`-like function. The functions
            // `check_addr_of_expr` and `check_into_iter_call_arg` determine whether the call is unnecessary
            // based on its context, that is, whether it is a referent in an `AddrOf` expression, an
            // argument in a `into_iter` call, or an argument in the call of some other function.
            if check_addr_of_expr(cx, expr, method_name, method_def_id, receiver) {
                return;
            }
            if check_into_iter_call_arg(cx, expr, method_name, receiver, msrv) {
                return;
            }
            if check_string_from_utf8(cx, expr, receiver) {
                return;
            }
            check_other_call_arg(cx, expr, method_name, receiver);
        }
    } else {
        check_borrow_predicate(cx, expr);
    }
}

/// Checks whether `expr` is a referent in an `AddrOf` expression and, if so, determines whether its
/// call of a `to_owned`-like function is unnecessary.
#[allow(clippy::too_many_lines)]
fn check_addr_of_expr(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    method_name: Symbol,
    method_def_id: DefId,
    receiver: &Expr<'_>,
) -> bool {
    if let Some(parent) = get_parent_expr(cx, expr)
        && let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, _) = parent.kind
        && let adjustments = cx.typeck_results().expr_adjustments(parent).iter().collect::<Vec<_>>()
        && let
            // For matching uses of `Cow::from`
            [
                Adjustment {
                    kind: Adjust::Deref(None),
                    target: referent_ty,
                },
                Adjustment {
                    kind: Adjust::Borrow(_),
                    target: target_ty,
                },
            ]
            // For matching uses of arrays
            | [
                Adjustment {
                    kind: Adjust::Deref(None),
                    target: referent_ty,
                },
                Adjustment {
                    kind: Adjust::Borrow(_),
                    ..
                },
                Adjustment {
                    kind: Adjust::Pointer(_),
                    target: target_ty,
                },
            ]
            // For matching everything else
            | [
                Adjustment {
                    kind: Adjust::Deref(None),
                    target: referent_ty,
                },
                Adjustment {
                    kind: Adjust::Deref(Some(OverloadedDeref { .. })),
                    ..
                },
                Adjustment {
                    kind: Adjust::Borrow(_),
                    target: target_ty,
                },
            ] = adjustments[..]
        && let receiver_ty = cx.typeck_results().expr_ty(receiver)
        && let (target_ty, n_target_refs) = peel_middle_ty_refs(*target_ty)
        && let (receiver_ty, n_receiver_refs) = peel_middle_ty_refs(receiver_ty)
        // Only flag cases satisfying at least one of the following three conditions:
        // * the referent and receiver types are distinct
        // * the referent/receiver type is a copyable array
        // * the method is `Cow::into_owned`
        // This restriction is to ensure there is no overlap between `redundant_clone` and this
        // lint. It also avoids the following false positive:
        //  https://github.com/rust-lang/rust-clippy/issues/8759
        //   Arrays are a bit of a corner case. Non-copyable arrays are handled by
        // `redundant_clone`, but copyable arrays are not.
        && (*referent_ty != receiver_ty
            || (matches!(referent_ty.kind(), ty::Array(..)) && is_copy(cx, *referent_ty))
            || is_cow_into_owned(cx, method_name, method_def_id))
        && let Some(receiver_snippet) = receiver.span.get_source_text(cx)
    {
        if receiver_ty == target_ty && n_target_refs >= n_receiver_refs {
            span_lint_and_sugg(
                cx,
                UNNECESSARY_TO_OWNED,
                parent.span,
                format!("unnecessary use of `{method_name}`"),
                "use",
                format!(
                    "{:&>width$}{receiver_snippet}",
                    "",
                    width = n_target_refs - n_receiver_refs
                ),
                Applicability::MachineApplicable,
            );
            return true;
        }
        if let Some(deref_trait_id) = cx.tcx.get_diagnostic_item(sym::Deref)
            && implements_trait(cx, receiver_ty, deref_trait_id, &[])
            && cx.get_associated_type(receiver_ty, deref_trait_id, sym::Target) == Some(target_ty)
            // Make sure that it's actually calling the right `.to_string()`, (#10033)
            // *or* this is a `Cow::into_owned()` call (which would be the wrong into_owned receiver (str != Cow)
            // but that's ok for Cow::into_owned specifically)
            && (cx.typeck_results().expr_ty_adjusted(receiver).peel_refs() == target_ty
                || is_cow_into_owned(cx, method_name, method_def_id))
        {
            if n_receiver_refs > 0 {
                span_lint_and_sugg(
                    cx,
                    UNNECESSARY_TO_OWNED,
                    parent.span,
                    format!("unnecessary use of `{method_name}`"),
                    "use",
                    receiver_snippet.to_owned(),
                    Applicability::MachineApplicable,
                );
            } else {
                span_lint_and_sugg(
                    cx,
                    UNNECESSARY_TO_OWNED,
                    expr.span.with_lo(receiver.span.hi()),
                    format!("unnecessary use of `{method_name}`"),
                    "remove this",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }
            return true;
        }
        if let Some(as_ref_trait_id) = cx.tcx.get_diagnostic_item(sym::AsRef)
            && implements_trait(cx, receiver_ty, as_ref_trait_id, &[GenericArg::from(target_ty)])
        {
            span_lint_and_sugg(
                cx,
                UNNECESSARY_TO_OWNED,
                parent.span,
                format!("unnecessary use of `{method_name}`"),
                "use",
                format!("{receiver_snippet}.as_ref()"),
                Applicability::MachineApplicable,
            );
            return true;
        }
    }
    false
}

/// Checks whether `expr` is an argument in an `into_iter` call and, if so, determines whether its
/// call of a `to_owned`-like function is unnecessary.
fn check_into_iter_call_arg(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    method_name: Symbol,
    receiver: &Expr<'_>,
    msrv: Msrv,
) -> bool {
    if let Some(parent) = get_parent_expr(cx, expr)
        && let Some(callee_def_id) = fn_def_id(cx, parent)
        && is_into_iter(cx, callee_def_id)
        && let Some(iterator_trait_id) = cx.tcx.get_diagnostic_item(sym::Iterator)
        && let parent_ty = cx.typeck_results().expr_ty(parent)
        && implements_trait(cx, parent_ty, iterator_trait_id, &[])
        && let Some(item_ty) = get_iterator_item_ty(cx, parent_ty)
        && let Some(receiver_snippet) = receiver.span.get_source_text(cx)
        // If the receiver is a `Cow`, we can't remove the `into_owned` generally, see https://github.com/rust-lang/rust-clippy/issues/13624.
        && !is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(receiver), sym::Cow)
        // Calling `iter()` on a temporary object can lead to false positives. #14242
        && !is_expr_temporary_value(cx, receiver)
    {
        if unnecessary_iter_cloned::check_for_loop_iter(cx, parent, method_name, receiver, true) {
            return true;
        }

        let cloned_or_copied = if is_copy(cx, item_ty) && msrv.meets(cx, msrvs::ITERATOR_COPIED) {
            "copied"
        } else {
            "cloned"
        };
        // The next suggestion may be incorrect because the removal of the `to_owned`-like
        // function could cause the iterator to hold a reference to a resource that is used
        // mutably. See https://github.com/rust-lang/rust-clippy/issues/8148.
        span_lint_and_sugg(
            cx,
            UNNECESSARY_TO_OWNED,
            parent.span,
            format!("unnecessary use of `{method_name}`"),
            "use",
            format!("{receiver_snippet}.iter().{cloned_or_copied}()"),
            Applicability::MaybeIncorrect,
        );
        return true;
    }
    false
}

/// Checks for `&String::from_utf8(bytes.{to_vec,to_owned,...}()).unwrap()` coercing to `&str`,
/// which can be written as just `std::str::from_utf8(bytes).unwrap()`.
fn check_string_from_utf8<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, receiver: &'tcx Expr<'tcx>) -> bool {
    if let Some((call, arg)) = skip_addr_of_ancestors(cx, expr)
        && !arg.span.from_expansion()
        && let ExprKind::Call(callee, _) = call.kind
        && fn_def_id(cx, call).is_some_and(|did| cx.tcx.is_diagnostic_item(sym::string_from_utf8, did))
        && let Some(unwrap_call) = get_parent_expr(cx, call)
        && let ExprKind::MethodCall(unwrap_method_name, ..) = unwrap_call.kind
        && matches!(unwrap_method_name.ident.name, sym::unwrap | sym::expect)
        && let Some(ref_string) = get_parent_expr(cx, unwrap_call)
        && let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, _) = ref_string.kind
        && let adjusted_ty = cx.typeck_results().expr_ty_adjusted(ref_string)
        // `&...` creates a `&String`, so only actually lint if this coerces to a `&str`
        && matches!(adjusted_ty.kind(), ty::Ref(_, ty, _) if ty.is_str())
    {
        span_lint_and_then(
            cx,
            UNNECESSARY_TO_OWNED,
            ref_string.span,
            "allocating a new `String` only to create a temporary `&str` from it",
            |diag| {
                let arg_suggestion = format!(
                    "{borrow}{recv_snippet}",
                    recv_snippet = snippet(cx, receiver.span.source_callsite(), ".."),
                    borrow = if cx.typeck_results().expr_ty(receiver).is_ref() {
                        ""
                    } else {
                        // If not already a reference, prefix with a borrow so that it can coerce to one
                        "&"
                    }
                );

                diag.multipart_suggestion(
                    "convert from `&[u8]` to `&str` directly",
                    vec![
                        // `&String::from_utf8(bytes.to_vec()).unwrap()`
                        //   ^^^^^^^^^^^^^^^^^
                        (callee.span, "core::str::from_utf8".into()),
                        // `&String::from_utf8(bytes.to_vec()).unwrap()`
                        //  ^
                        (
                            ref_string.span.shrink_to_lo().to(unwrap_call.span.shrink_to_lo()),
                            String::new(),
                        ),
                        // `&String::from_utf8(bytes.to_vec()).unwrap()`
                        //                     ^^^^^^^^^^^^^^
                        (arg.span, arg_suggestion),
                    ],
                    Applicability::MachineApplicable,
                );
            },
        );
        true
    } else {
        false
    }
}

/// Checks whether `expr` is an argument in an `into_iter` call and, if so, determines whether its
/// call of a `to_owned`-like function is unnecessary.
fn check_split_call_arg(cx: &LateContext<'_>, expr: &Expr<'_>, method_name: Symbol, receiver: &Expr<'_>) -> bool {
    if let Some(parent) = get_parent_expr(cx, expr)
        && let Some((sym::split, argument_expr)) = get_fn_name_and_arg(cx, parent)
        && let Some(receiver_snippet) = receiver.span.get_source_text(cx)
        && let Some(arg_snippet) = argument_expr.span.get_source_text(cx)
    {
        // We may end-up here because of an expression like `x.to_string().split(…)` where the type of `x`
        // implements `AsRef<str>` but does not implement `Deref<Target = str>`. In this case, we have to
        // add `.as_ref()` to the suggestion.
        let as_ref = if is_type_lang_item(cx, cx.typeck_results().expr_ty(expr), LangItem::String)
            && let Some(deref_trait_id) = cx.tcx.get_diagnostic_item(sym::Deref)
            && cx.get_associated_type(cx.typeck_results().expr_ty(receiver), deref_trait_id, sym::Target)
                != Some(cx.tcx.types.str_)
        {
            ".as_ref()"
        } else {
            ""
        };

        // The next suggestion may be incorrect because the removal of the `to_owned`-like
        // function could cause the iterator to hold a reference to a resource that is used
        // mutably. See https://github.com/rust-lang/rust-clippy/issues/8148.
        span_lint_and_sugg(
            cx,
            UNNECESSARY_TO_OWNED,
            parent.span,
            format!("unnecessary use of `{method_name}`"),
            "use",
            format!("{receiver_snippet}{as_ref}.split({arg_snippet})"),
            Applicability::MaybeIncorrect,
        );
        return true;
    }
    false
}

fn get_fn_name_and_arg<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> Option<(Symbol, Expr<'tcx>)> {
    match &expr.kind {
        ExprKind::MethodCall(path, _, [arg_expr], ..) => Some((path.ident.name, *arg_expr)),
        ExprKind::Call(
            Expr {
                kind: ExprKind::Path(qpath),
                hir_id: path_hir_id,
                ..
            },
            [arg_expr],
        ) => {
            // Only return Fn-like DefIds, not the DefIds of statics/consts/etc that contain or
            // deref to fn pointers, dyn Fn, impl Fn - #8850
            if let Res::Def(DefKind::Fn | DefKind::Ctor(..) | DefKind::AssocFn, def_id) =
                cx.typeck_results().qpath_res(qpath, *path_hir_id)
                && let Some(fn_name) = cx.tcx.opt_item_name(def_id)
            {
                Some((fn_name, *arg_expr))
            } else {
                None
            }
        },
        _ => None,
    }
}

/// Checks whether `expr` is an argument in a function call and, if so, determines whether its call
/// of a `to_owned`-like function is unnecessary.
fn check_other_call_arg<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    method_name: Symbol,
    receiver: &'tcx Expr<'tcx>,
) -> bool {
    if let Some((maybe_call, maybe_arg)) = skip_addr_of_ancestors(cx, expr)
        && let Some((callee_def_id, _, recv, call_args)) = get_callee_generic_args_and_args(cx, maybe_call)
        && let fn_sig = cx.tcx.fn_sig(callee_def_id).instantiate_identity().skip_binder()
        && let Some(i) = recv.into_iter().chain(call_args).position(|arg| arg.hir_id == maybe_arg.hir_id)
        && let Some(input) = fn_sig.inputs().get(i)
        && let (input, n_refs) = peel_middle_ty_refs(*input)
        && let (trait_predicates, _) = get_input_traits_and_projections(cx, callee_def_id, input)
        && let Some(sized_def_id) = cx.tcx.lang_items().sized_trait()
        && let Some(meta_sized_def_id) = cx.tcx.lang_items().meta_sized_trait()
        && let [trait_predicate] = trait_predicates
            .iter()
            .filter(|trait_predicate| trait_predicate.def_id() != sized_def_id)
            .filter(|trait_predicate| trait_predicate.def_id() != meta_sized_def_id)
            .collect::<Vec<_>>()[..]
        && let Some(deref_trait_id) = cx.tcx.get_diagnostic_item(sym::Deref)
        && let Some(as_ref_trait_id) = cx.tcx.get_diagnostic_item(sym::AsRef)
        && (trait_predicate.def_id() == deref_trait_id || trait_predicate.def_id() == as_ref_trait_id)
        && let receiver_ty = cx.typeck_results().expr_ty(receiver)
        // We can't add an `&` when the trait is `Deref` because `Target = &T` won't match
        // `Target = T`.
        && let Some((n_refs, receiver_ty)) = if n_refs > 0 || is_copy(cx, receiver_ty) {
            Some((n_refs, receiver_ty))
        } else if trait_predicate.def_id() != deref_trait_id {
            Some((1, Ty::new_imm_ref(cx.tcx,
                cx.tcx.lifetimes.re_erased,
                receiver_ty,
            )))
        } else {
            None
        }
        && can_change_type(cx, maybe_arg, receiver_ty)
        && let Some(receiver_snippet) = receiver.span.get_source_text(cx)
    {
        span_lint_and_sugg(
            cx,
            UNNECESSARY_TO_OWNED,
            maybe_arg.span,
            format!("unnecessary use of `{method_name}`"),
            "use",
            format!("{:&>n_refs$}{receiver_snippet}", ""),
            Applicability::MachineApplicable,
        );
        return true;
    }
    false
}

/// Walks an expression's ancestors until it finds a non-`AddrOf` expression. Returns the first such
/// expression found (if any) along with the immediately prior expression.
fn skip_addr_of_ancestors<'tcx>(
    cx: &LateContext<'tcx>,
    mut expr: &'tcx Expr<'tcx>,
) -> Option<(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>)> {
    while let Some(parent) = get_parent_expr(cx, expr) {
        if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, _) = parent.kind {
            expr = parent;
        } else {
            return Some((parent, expr));
        }
    }
    None
}

/// Checks whether an expression is a function or method call and, if so, returns its `DefId`,
/// `GenericArgs`, and arguments.
fn get_callee_generic_args_and_args<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
) -> Option<(
    DefId,
    GenericArgsRef<'tcx>,
    Option<&'tcx Expr<'tcx>>,
    &'tcx [Expr<'tcx>],
)> {
    if let ExprKind::Call(callee, args) = expr.kind
        && let callee_ty = cx.typeck_results().expr_ty(callee)
        && let ty::FnDef(callee_def_id, _) = callee_ty.kind()
    {
        let generic_args = cx.typeck_results().node_args(callee.hir_id);
        return Some((*callee_def_id, generic_args, None, args));
    }
    if let ExprKind::MethodCall(_, recv, args, _) = expr.kind
        && let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
    {
        let generic_args = cx.typeck_results().node_args(expr.hir_id);
        return Some((method_def_id, generic_args, Some(recv), args));
    }
    None
}

/// Returns the `TraitPredicate`s and `ProjectionPredicate`s for a function's input type.
fn get_input_traits_and_projections<'tcx>(
    cx: &LateContext<'tcx>,
    callee_def_id: DefId,
    input: Ty<'tcx>,
) -> (Vec<TraitPredicate<'tcx>>, Vec<ProjectionPredicate<'tcx>>) {
    let mut trait_predicates = Vec::new();
    let mut projection_predicates = Vec::new();
    for predicate in cx.tcx.param_env(callee_def_id).caller_bounds() {
        match predicate.kind().skip_binder() {
            ClauseKind::Trait(trait_predicate) => {
                if trait_predicate.trait_ref.self_ty() == input {
                    trait_predicates.push(trait_predicate);
                }
            },
            ClauseKind::Projection(projection_predicate) => {
                if projection_predicate.projection_term.self_ty() == input {
                    projection_predicates.push(projection_predicate);
                }
            },
            _ => {},
        }
    }
    (trait_predicates, projection_predicates)
}

#[expect(clippy::too_many_lines)]
fn can_change_type<'a>(cx: &LateContext<'a>, mut expr: &'a Expr<'a>, mut ty: Ty<'a>) -> bool {
    for (_, node) in cx.tcx.hir_parent_iter(expr.hir_id) {
        match node {
            Node::Stmt(_) => return true,
            Node::Block(..) => {},
            Node::Item(item) => {
                if let ItemKind::Fn { body: body_id, .. } = &item.kind
                    && let output_ty = return_ty(cx, item.owner_id)
                    && rustc_hir_typeck::can_coerce(cx.tcx, cx.param_env, item.owner_id.def_id, ty, output_ty)
                {
                    if has_lifetime(output_ty) && has_lifetime(ty) {
                        return false;
                    }
                    let body = cx.tcx.hir_body(*body_id);
                    let body_expr = &body.value;
                    let mut count = 0;
                    return find_all_ret_expressions(cx, body_expr, |_| {
                        count += 1;
                        count <= 1
                    });
                }
            },
            Node::Expr(parent_expr) => {
                if let Some((callee_def_id, call_generic_args, recv, call_args)) =
                    get_callee_generic_args_and_args(cx, parent_expr)
                {
                    let bound_fn_sig = cx.tcx.fn_sig(callee_def_id);
                    let fn_sig = bound_fn_sig.skip_binder();
                    if let Some(arg_index) = recv
                        .into_iter()
                        .chain(call_args)
                        .position(|arg| arg.hir_id == expr.hir_id)
                        && let param_ty = fn_sig.input(arg_index).skip_binder()
                        && let ty::Param(ParamTy { index: param_index , ..}) = *param_ty.kind()
                        // https://github.com/rust-lang/rust-clippy/issues/9504 and https://github.com/rust-lang/rust-clippy/issues/10021
                        && (param_index as usize) < call_generic_args.len()
                    {
                        if fn_sig
                            .skip_binder()
                            .inputs()
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != arg_index)
                            .any(|(_, ty)| ty.contains(param_ty))
                        {
                            return false;
                        }

                        let mut trait_predicates =
                            cx.tcx
                                .param_env(callee_def_id)
                                .caller_bounds()
                                .iter()
                                .filter(|predicate| {
                                    if let ClauseKind::Trait(trait_predicate) = predicate.kind().skip_binder()
                                        && trait_predicate.trait_ref.self_ty() == param_ty
                                    {
                                        true
                                    } else {
                                        false
                                    }
                                });

                        let new_subst = cx
                            .tcx
                            .mk_args_from_iter(call_generic_args.iter().enumerate().map(|(i, t)| {
                                if i == param_index as usize {
                                    GenericArg::from(ty)
                                } else {
                                    t
                                }
                            }));

                        if trait_predicates.any(|predicate| {
                            let predicate = bound_fn_sig.rebind(predicate).instantiate(cx.tcx, new_subst);
                            let obligation = Obligation::new(cx.tcx, ObligationCause::dummy(), cx.param_env, predicate);
                            !cx.tcx
                                .infer_ctxt()
                                .build(cx.typing_mode())
                                .predicate_must_hold_modulo_regions(&obligation)
                        }) {
                            return false;
                        }

                        let output_ty = cx.tcx.instantiate_bound_regions_with_erased(fn_sig.output());
                        if output_ty.contains(param_ty) {
                            if let Ok(new_ty) = cx.tcx.try_instantiate_and_normalize_erasing_regions(
                                new_subst,
                                cx.typing_env(),
                                bound_fn_sig.rebind(output_ty),
                            ) {
                                expr = parent_expr;
                                ty = new_ty;
                                continue;
                            }
                            return false;
                        }

                        return true;
                    }
                } else if let ExprKind::Block(..) = parent_expr.kind {
                    continue;
                }
                return false;
            },
            _ => return false,
        }
    }

    false
}

fn has_lifetime(ty: Ty<'_>) -> bool {
    ty.walk().any(|t| matches!(t.kind(), GenericArgKind::Lifetime(_)))
}

/// Returns true if the named method is `Iterator::cloned` or `Iterator::copied`.
fn is_cloned_or_copied(cx: &LateContext<'_>, method_name: Symbol, method_def_id: DefId) -> bool {
    matches!(method_name, sym::cloned | sym::copied) && is_diag_trait_item(cx, method_def_id, sym::Iterator)
}

/// Returns true if the named method can be used to convert the receiver to its "owned"
/// representation.
fn is_to_owned_like<'a>(cx: &LateContext<'a>, call_expr: &Expr<'a>, method_name: Symbol, method_def_id: DefId) -> bool {
    is_cow_into_owned(cx, method_name, method_def_id)
        || (method_name != sym::to_string && is_clone_like(cx, method_name, method_def_id))
        || is_to_string_on_string_like(cx, call_expr, method_name, method_def_id)
}

/// Returns true if the named method is `Cow::into_owned`.
fn is_cow_into_owned(cx: &LateContext<'_>, method_name: Symbol, method_def_id: DefId) -> bool {
    method_name == sym::into_owned && is_diag_item_method(cx, method_def_id, sym::Cow)
}

/// Returns true if the named method is `ToString::to_string` and it's called on a type that
/// is string-like i.e. implements `AsRef<str>` or `Deref<Target = str>`.
fn is_to_string_on_string_like<'a>(
    cx: &LateContext<'_>,
    call_expr: &'a Expr<'a>,
    method_name: Symbol,
    method_def_id: DefId,
) -> bool {
    if method_name != sym::to_string || !is_diag_trait_item(cx, method_def_id, sym::ToString) {
        return false;
    }

    if let Some(args) = cx.typeck_results().node_args_opt(call_expr.hir_id)
        && let [generic_arg] = args.as_slice()
        && let GenericArgKind::Type(ty) = generic_arg.kind()
        && let Some(deref_trait_id) = cx.tcx.get_diagnostic_item(sym::Deref)
        && let Some(as_ref_trait_id) = cx.tcx.get_diagnostic_item(sym::AsRef)
        && (cx.get_associated_type(ty, deref_trait_id, sym::Target) == Some(cx.tcx.types.str_)
            || implements_trait(cx, ty, as_ref_trait_id, &[cx.tcx.types.str_.into()]))
    {
        true
    } else {
        false
    }
}

fn std_map_key<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    match ty.kind() {
        ty::Adt(adt, args)
            if matches!(
                cx.tcx.get_diagnostic_name(adt.did()),
                Some(sym::BTreeMap | sym::BTreeSet | sym::HashMap | sym::HashSet)
            ) =>
        {
            Some(args.type_at(0))
        },
        _ => None,
    }
}

fn is_str_and_string(cx: &LateContext<'_>, arg_ty: Ty<'_>, original_arg_ty: Ty<'_>) -> bool {
    original_arg_ty.is_str() && is_type_lang_item(cx, arg_ty, LangItem::String)
}

fn is_slice_and_vec(cx: &LateContext<'_>, arg_ty: Ty<'_>, original_arg_ty: Ty<'_>) -> bool {
    (original_arg_ty.is_slice() || original_arg_ty.is_array() || original_arg_ty.is_array_slice())
        && is_type_diagnostic_item(cx, arg_ty, sym::Vec)
}

// This function will check the following:
// 1. The argument is a non-mutable reference.
// 2. It calls `to_owned()`, `to_string()` or `to_vec()`.
// 3. That the method is called on `String` or on `Vec` (only types supported for the moment).
fn check_if_applicable_to_argument<'tcx>(cx: &LateContext<'tcx>, arg: &Expr<'tcx>) {
    if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, expr) = arg.kind
        && let ExprKind::MethodCall(method_path, caller, &[], _) = expr.kind
        && let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && let method_name = method_path.ident.name
        && match method_name {
            sym::to_owned => cx.tcx.is_diagnostic_item(sym::to_owned_method, method_def_id),
            sym::to_string => cx.tcx.is_diagnostic_item(sym::to_string_method, method_def_id),
            sym::to_vec => cx
                .tcx
                .impl_of_assoc(method_def_id)
                .filter(|&impl_did| cx.tcx.type_of(impl_did).instantiate_identity().is_slice())
                .is_some(),
            _ => false,
        }
        && let original_arg_ty = cx.typeck_results().node_type(caller.hir_id).peel_refs()
        && let arg_ty = cx.typeck_results().expr_ty(arg)
        && let ty::Ref(_, arg_ty, Mutability::Not) = arg_ty.kind()
        // FIXME: try to fix `can_change_type` to make it work in this case.
        // && can_change_type(cx, caller, *arg_ty)
        && let arg_ty = arg_ty.peel_refs()
        // For now we limit this lint to `String` and `Vec`.
        && (is_str_and_string(cx, arg_ty, original_arg_ty) || is_slice_and_vec(cx, arg_ty, original_arg_ty))
        && let Some(snippet) = caller.span.get_source_text(cx)
    {
        span_lint_and_sugg(
            cx,
            UNNECESSARY_TO_OWNED,
            arg.span,
            format!("unnecessary use of `{method_name}`"),
            "replace it with",
            if original_arg_ty.is_array() {
                format!("{snippet}.as_slice()")
            } else {
                snippet.to_owned()
            },
            Applicability::MaybeIncorrect,
        );
    }
}

// In std "map types", the getters all expect a `Borrow<Key>` generic argument. So in here, we
// check that:
// 1. This is a method with only one argument that doesn't come from a trait.
// 2. That it has `Borrow` in its generic predicates.
// 3. `Self` is a std "map type" (ie `HashSet`, `HashMap`, `BTreeSet`, `BTreeMap`).
// 4. The key to the "map type" is not a reference.
fn check_borrow_predicate<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
    if let ExprKind::MethodCall(_, caller, &[arg], _) = expr.kind
        && let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && cx.tcx.trait_of_assoc(method_def_id).is_none()
        && let Some(borrow_id) = cx.tcx.get_diagnostic_item(sym::Borrow)
        && cx.tcx.predicates_of(method_def_id).predicates.iter().any(|(pred, _)| {
            if let ClauseKind::Trait(trait_pred) = pred.kind().skip_binder()
                && trait_pred.polarity == ty::PredicatePolarity::Positive
                && trait_pred.trait_ref.def_id == borrow_id
            {
                true
            } else {
                false
            }
        })
        && let caller_ty = cx.typeck_results().expr_ty(caller)
        // For now we limit it to "map types".
        && let Some(key_ty) = std_map_key(cx, caller_ty)
        // We need to check that the key type is not a reference.
        && !key_ty.is_ref()
    {
        check_if_applicable_to_argument(cx, &arg);
    }
}
