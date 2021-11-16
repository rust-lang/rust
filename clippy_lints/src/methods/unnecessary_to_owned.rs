use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::{implements_trait, is_copy, peel_mid_ty_refs};
use clippy_utils::{get_parent_expr, match_def_path, paths};
use rustc_errors::Applicability;
use rustc_hir::{def_id::DefId, BorrowKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::mir::Mutability;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, OverloadedDeref};
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, SubstsRef};
use rustc_middle::ty::{self, PredicateKind, ProjectionPredicate, TraitPredicate, Ty};
use rustc_span::{sym, Symbol};
use std::cmp::max;

use super::UNNECESSARY_TO_OWNED;

const TO_OWNED_LIKE_PATHS: &[&[&str]] = &[
    &paths::COW_INTO_OWNED,
    &paths::OS_STR_TO_OS_STRING,
    &paths::PATH_TO_PATH_BUF,
    &paths::SLICE_TO_VEC,
    &paths::TO_OWNED_METHOD,
    &paths::TO_STRING_METHOD,
];

pub fn check(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, method_name: Symbol, args: &'tcx [Expr<'tcx>]) {
    if_chain! {
        if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if TO_OWNED_LIKE_PATHS
            .iter()
            .any(|path| match_def_path(cx, method_def_id, path));
        if let [receiver] = args;
        then {
            // At this point, we know the call is of a `to_owned`-like function. The functions
            // `check_addr_of_expr` and `check_call_arg` determine whether the call is unnecessary
            // based on its context, that is, whether it is a referent in an `AddrOf` expression or
            // an argument in a function call.
            if check_addr_of_expr(cx, expr, method_name, receiver) {
                return;
            }
            check_call_arg(cx, expr, method_name, receiver);
        }
    }
}

/// Checks whether `expr` is a referent in an `AddrOf` expression and, if so, determines whether its
/// call of a `to_owned`-like function is unnecessary.
fn check_addr_of_expr(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    method_name: Symbol,
    receiver: &'tcx Expr<'tcx>,
) -> bool {
    if_chain! {
        if let Some(parent) = get_parent_expr(cx, expr);
        if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, _) = parent.kind;
        let adjustments = cx.typeck_results().expr_adjustments(parent).iter().collect::<Vec<_>>();
        if let Some(target_ty) = match adjustments[..]
        {
            // For matching uses of `Cow::from`
            [
                Adjustment {
                    kind: Adjust::Deref(None),
                    ..
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
                    ..
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
                    ..
                },
                Adjustment {
                    kind: Adjust::Deref(Some(OverloadedDeref { .. })),
                    ..
                },
                Adjustment {
                    kind: Adjust::Borrow(_),
                    target: target_ty,
                },
            ] => Some(target_ty),
            _ => None,
        };
        then {
            let (target_ty, n_target_refs) = peel_mid_ty_refs(target_ty);
            let receiver_ty = cx.typeck_results().expr_ty(receiver);
            let (receiver_ty, n_receiver_refs) = peel_mid_ty_refs(receiver_ty);
            if_chain! {
                if receiver_ty == target_ty;
                if n_target_refs >= n_receiver_refs;
                if let Some(receiver_snippet) = snippet_opt(cx, receiver.span);
                then {
                    span_lint_and_sugg(
                        cx,
                        UNNECESSARY_TO_OWNED,
                        parent.span,
                        &format!("unnecessary use of `{}`", method_name),
                        "use",
                        format!("{:&>width$}{}", "", receiver_snippet, width = n_target_refs - n_receiver_refs),
                        Applicability::MachineApplicable,
                    );
                    return true;
                }
            }
            if implements_deref_trait(cx, receiver_ty, target_ty) {
                span_lint_and_sugg(
                    cx,
                    UNNECESSARY_TO_OWNED,
                    expr.span.with_lo(receiver.span.hi()),
                    &format!("unnecessary use of `{}`", method_name),
                    "remove this",
                    String::new(),
                    Applicability::MachineApplicable,
                );
                return true;
            }
            if_chain! {
                if let Some(as_ref_trait_id) = cx.tcx.get_diagnostic_item(sym::AsRef);
                if implements_trait(cx, receiver_ty, as_ref_trait_id, &[GenericArg::from(target_ty)]);
                if let Some(receiver_snippet) = snippet_opt(cx, receiver.span);
                then {
                    span_lint_and_sugg(
                        cx,
                        UNNECESSARY_TO_OWNED,
                        parent.span,
                        &format!("unnecessary use of `{}`", method_name),
                        "use",
                        format!("{}.as_ref()", receiver_snippet),
                        Applicability::MachineApplicable,
                    );
                    return true;
                }
            }
        }
    }
    false
}

/// Checks whether `expr` is an argument in a function call and, if so, determines whether its call
/// of a `to_owned`-like function is unnecessary.
fn check_call_arg(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, method_name: Symbol, receiver: &'tcx Expr<'tcx>) {
    if_chain! {
        if let Some((maybe_call, maybe_arg)) = skip_addr_of_ancestors(cx, expr);
        if let Some((callee_def_id, call_substs, call_args)) = get_callee_substs_and_args(cx, maybe_call);
        let fn_sig = cx.tcx.fn_sig(callee_def_id).skip_binder();
        if let Some(i) = call_args.iter().position(|arg| arg.hir_id == maybe_arg.hir_id);
        if let Some(input) = fn_sig.inputs().get(i);
        let (input, n_refs) = peel_mid_ty_refs(input);
        if let (trait_predicates, projection_predicates) = get_input_traits_and_projections(cx, callee_def_id, input);
        if let Some(sized_def_id) = cx.tcx.lang_items().sized_trait();
        if let [trait_predicate] = trait_predicates
            .iter()
            .filter(|trait_predicate| trait_predicate.def_id() != sized_def_id)
            .collect::<Vec<_>>()[..];
        if let Some(deref_trait_id) = cx.tcx.get_diagnostic_item(sym::Deref);
        if let Some(as_ref_trait_id) = cx.tcx.get_diagnostic_item(sym::AsRef);
        let receiver_ty = cx.typeck_results().expr_ty(receiver);
        // If the callee has type parameters, they could appear in `projection_predicate.ty` or the
        // types of `trait_predicate.trait_ref.substs`.
        if if trait_predicate.def_id() == deref_trait_id {
            if let [projection_predicate] = projection_predicates[..] {
                let normalized_ty =
                    cx.tcx.subst_and_normalize_erasing_regions(call_substs, cx.param_env, projection_predicate.ty);
                implements_deref_trait(cx, receiver_ty, normalized_ty)
            } else {
                false
            }
        } else if trait_predicate.def_id() == as_ref_trait_id {
            let composed_substs = compose_substs(
                cx,
                &trait_predicate.trait_ref.substs.iter().skip(1).collect::<Vec<_>>()[..],
                call_substs
            );
            implements_trait(cx, receiver_ty, as_ref_trait_id, &composed_substs)
        } else {
            false
        };
        // We can't add an `&` when the trait is `Deref` because `Target = &T` won't match
        // `Target = T`.
        if n_refs > 0 || is_copy(cx, receiver_ty) || trait_predicate.def_id() != deref_trait_id;
        let n_refs = max(n_refs, if is_copy(cx, receiver_ty) { 0 } else { 1 });
        if let Some(receiver_snippet) = snippet_opt(cx, receiver.span);
        then {
            span_lint_and_sugg(
                cx,
                UNNECESSARY_TO_OWNED,
                maybe_arg.span,
                &format!("unnecessary use of `{}`", method_name),
                "use",
                format!("{:&>width$}{}", "", receiver_snippet, width = n_refs),
                Applicability::MachineApplicable,
            );
        }
    }
}

/// Walks an expression's ancestors until it finds a non-`AddrOf` expression. Returns the first such
/// expression found (if any) along with the immediately prior expression.
fn skip_addr_of_ancestors(
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
/// `Substs`, and arguments.
fn get_callee_substs_and_args(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
) -> Option<(DefId, SubstsRef<'tcx>, &'tcx [Expr<'tcx>])> {
    if_chain! {
        if let ExprKind::Call(callee, args) = expr.kind;
        let callee_ty = cx.typeck_results().expr_ty(callee);
        if let ty::FnDef(callee_def_id, _) = callee_ty.kind();
        then {
            let substs = cx.typeck_results().node_substs(callee.hir_id);
            return Some((*callee_def_id, substs, args));
        }
    }
    if_chain! {
        if let ExprKind::MethodCall(_, _, args, _) = expr.kind;
        if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        then {
            let substs = cx.typeck_results().node_substs(expr.hir_id);
            return Some((method_def_id, substs, args));
        }
    }
    None
}

/// Returns the `TraitPredicate`s and `ProjectionPredicate`s for a function's input type.
fn get_input_traits_and_projections(
    cx: &LateContext<'tcx>,
    callee_def_id: DefId,
    input: Ty<'tcx>,
) -> (Vec<TraitPredicate<'tcx>>, Vec<ProjectionPredicate<'tcx>>) {
    let mut trait_predicates = Vec::new();
    let mut projection_predicates = Vec::new();
    for (predicate, _) in cx.tcx.predicates_of(callee_def_id).predicates.iter() {
        // `substs` should have 1 + n elements. The first is the type on the left hand side of an
        // `as`. The remaining n are trait parameters.
        let is_input_substs = |substs: SubstsRef<'tcx>| {
            if_chain! {
                if let Some(arg) = substs.iter().next();
                if let GenericArgKind::Type(arg_ty) = arg.unpack();
                if arg_ty == input;
                then {
                    true
                } else {
                    false
                }
            }
        };
        match predicate.kind().skip_binder() {
            PredicateKind::Trait(trait_predicate) => {
                if is_input_substs(trait_predicate.trait_ref.substs) {
                    trait_predicates.push(trait_predicate);
                }
            },
            PredicateKind::Projection(projection_predicate) => {
                if is_input_substs(projection_predicate.projection_ty.substs) {
                    projection_predicates.push(projection_predicate);
                }
            },
            _ => {},
        }
    }
    (trait_predicates, projection_predicates)
}

/// Composes two substitutions by applying the latter to the types of the former.
fn compose_substs(cx: &LateContext<'tcx>, left: &[GenericArg<'tcx>], right: SubstsRef<'tcx>) -> Vec<GenericArg<'tcx>> {
    left.iter()
        .map(|arg| {
            if let GenericArgKind::Type(arg_ty) = arg.unpack() {
                let normalized_ty = cx.tcx.subst_and_normalize_erasing_regions(right, cx.param_env, arg_ty);
                GenericArg::from(normalized_ty)
            } else {
                *arg
            }
        })
        .collect()
}

/// Helper function to check whether a type implements the `Deref` trait.
fn implements_deref_trait(cx: &LateContext<'tcx>, ty: Ty<'tcx>, deref_target_ty: Ty<'tcx>) -> bool {
    if_chain! {
        if let Some(deref_trait_id) = cx.tcx.get_diagnostic_item(sym::Deref);
        if implements_trait(cx, ty, deref_trait_id, &[]);
        if let Some(deref_target_id) = cx.tcx.lang_items().deref_target();
        let substs = cx.tcx.mk_substs_trait(ty, &[]);
        let projection_ty = cx.tcx.mk_projection(deref_target_id, substs);
        let normalized_ty = cx.tcx.normalize_erasing_regions(cx.param_env, projection_ty);
        if normalized_ty == deref_target_ty;
        then {
            true
        } else {
            false
        }
    }
}
