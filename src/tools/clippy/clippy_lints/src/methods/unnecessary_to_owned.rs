use super::implicit_clone::is_clone_like;
use super::unnecessary_iter_cloned::{self, is_into_iter};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::{get_iterator_item_ty, implements_trait, is_copy, peel_mid_ty_refs};
use clippy_utils::visitors::find_all_ret_expressions;
use clippy_utils::{fn_def_id, get_parent_expr, is_diag_item_method, is_diag_trait_item, return_ty};
use rustc_errors::Applicability;
use rustc_hir::{def_id::DefId, BorrowKind, Expr, ExprKind, ItemKind, Node};
use rustc_hir_typeck::{FnCtxt, Inherited};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::LateContext;
use rustc_middle::mir::Mutability;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, OverloadedDeref};
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, SubstsRef};
use rustc_middle::ty::{self, ClauseKind, EarlyBinder, ParamTy, ProjectionPredicate, TraitPredicate, Ty};
use rustc_span::{sym, Symbol};
use rustc_trait_selection::traits::{query::evaluate_obligation::InferCtxtExt as _, Obligation, ObligationCause};

use super::UNNECESSARY_TO_OWNED;

pub fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    method_name: Symbol,
    receiver: &'tcx Expr<'_>,
    args: &'tcx [Expr<'_>],
    msrv: &Msrv,
) {
    if_chain! {
        if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if args.is_empty();
        then {
            if is_cloned_or_copied(cx, method_name, method_def_id) {
                unnecessary_iter_cloned::check(cx, expr, method_name, receiver);
            } else if is_to_owned_like(cx, expr, method_name, method_def_id) {
                // At this point, we know the call is of a `to_owned`-like function. The functions
                // `check_addr_of_expr` and `check_call_arg` determine whether the call is unnecessary
                // based on its context, that is, whether it is a referent in an `AddrOf` expression, an
                // argument in a `into_iter` call, or an argument in the call of some other function.
                if check_addr_of_expr(cx, expr, method_name, method_def_id, receiver) {
                    return;
                }
                if check_into_iter_call_arg(cx, expr, method_name, receiver, msrv) {
                    return;
                }
                check_other_call_arg(cx, expr, method_name, receiver);
            }
        }
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
    if_chain! {
        if let Some(parent) = get_parent_expr(cx, expr);
        if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, _) = parent.kind;
        let adjustments = cx.typeck_results().expr_adjustments(parent).iter().collect::<Vec<_>>();
        if let
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
            ] = adjustments[..];
        let receiver_ty = cx.typeck_results().expr_ty(receiver);
        let (target_ty, n_target_refs) = peel_mid_ty_refs(*target_ty);
        let (receiver_ty, n_receiver_refs) = peel_mid_ty_refs(receiver_ty);
        // Only flag cases satisfying at least one of the following three conditions:
        // * the referent and receiver types are distinct
        // * the referent/receiver type is a copyable array
        // * the method is `Cow::into_owned`
        // This restriction is to ensure there is no overlap between `redundant_clone` and this
        // lint. It also avoids the following false positive:
        //  https://github.com/rust-lang/rust-clippy/issues/8759
        //   Arrays are a bit of a corner case. Non-copyable arrays are handled by
        // `redundant_clone`, but copyable arrays are not.
        if *referent_ty != receiver_ty
            || (matches!(referent_ty.kind(), ty::Array(..)) && is_copy(cx, *referent_ty))
            || is_cow_into_owned(cx, method_name, method_def_id);
        if let Some(receiver_snippet) = snippet_opt(cx, receiver.span);
        then {
            if receiver_ty == target_ty && n_target_refs >= n_receiver_refs {
                span_lint_and_sugg(
                    cx,
                    UNNECESSARY_TO_OWNED,
                    parent.span,
                    &format!("unnecessary use of `{method_name}`"),
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
            if_chain! {
                if let Some(deref_trait_id) = cx.tcx.get_diagnostic_item(sym::Deref);
                if implements_trait(cx, receiver_ty, deref_trait_id, &[]);
                if cx.get_associated_type(receiver_ty, deref_trait_id, "Target") == Some(target_ty);
                // Make sure that it's actually calling the right `.to_string()`, (#10033)
                // *or* this is a `Cow::into_owned()` call (which would be the wrong into_owned receiver (str != Cow)
                // but that's ok for Cow::into_owned specifically)
                if cx.typeck_results().expr_ty_adjusted(receiver).peel_refs() == target_ty
                    || is_cow_into_owned(cx, method_name, method_def_id);
                then {
                    if n_receiver_refs > 0 {
                        span_lint_and_sugg(
                            cx,
                            UNNECESSARY_TO_OWNED,
                            parent.span,
                            &format!("unnecessary use of `{method_name}`"),
                            "use",
                            receiver_snippet,
                            Applicability::MachineApplicable,
                        );
                    } else {
                        span_lint_and_sugg(
                            cx,
                            UNNECESSARY_TO_OWNED,
                            expr.span.with_lo(receiver.span.hi()),
                            &format!("unnecessary use of `{method_name}`"),
                            "remove this",
                            String::new(),
                            Applicability::MachineApplicable,
                        );
                    }
                    return true;
                }
            }
            if_chain! {
                if let Some(as_ref_trait_id) = cx.tcx.get_diagnostic_item(sym::AsRef);
                if implements_trait(cx, receiver_ty, as_ref_trait_id, &[GenericArg::from(target_ty)]);
                then {
                    span_lint_and_sugg(
                        cx,
                        UNNECESSARY_TO_OWNED,
                        parent.span,
                        &format!("unnecessary use of `{method_name}`"),
                        "use",
                        format!("{receiver_snippet}.as_ref()"),
                        Applicability::MachineApplicable,
                    );
                    return true;
                }
            }
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
    msrv: &Msrv,
) -> bool {
    if_chain! {
        if let Some(parent) = get_parent_expr(cx, expr);
        if let Some(callee_def_id) = fn_def_id(cx, parent);
        if is_into_iter(cx, callee_def_id);
        if let Some(iterator_trait_id) = cx.tcx.get_diagnostic_item(sym::Iterator);
        let parent_ty = cx.typeck_results().expr_ty(parent);
        if implements_trait(cx, parent_ty, iterator_trait_id, &[]);
        if let Some(item_ty) = get_iterator_item_ty(cx, parent_ty);
        if let Some(receiver_snippet) = snippet_opt(cx, receiver.span);
        then {
            if unnecessary_iter_cloned::check_for_loop_iter(cx, parent, method_name, receiver, true) {
                return true;
            }
            let cloned_or_copied = if is_copy(cx, item_ty) && msrv.meets(msrvs::ITERATOR_COPIED) {
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
                &format!("unnecessary use of `{method_name}`"),
                "use",
                format!("{receiver_snippet}.iter().{cloned_or_copied}()"),
                Applicability::MaybeIncorrect,
            );
            return true;
        }
    }
    false
}

/// Checks whether `expr` is an argument in a function call and, if so, determines whether its call
/// of a `to_owned`-like function is unnecessary.
fn check_other_call_arg<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    method_name: Symbol,
    receiver: &'tcx Expr<'tcx>,
) -> bool {
    if_chain! {
        if let Some((maybe_call, maybe_arg)) = skip_addr_of_ancestors(cx, expr);
        if let Some((callee_def_id, _, recv, call_args)) = get_callee_substs_and_args(cx, maybe_call);
        let fn_sig = cx.tcx.fn_sig(callee_def_id).subst_identity().skip_binder();
        if let Some(i) = recv.into_iter().chain(call_args).position(|arg| arg.hir_id == maybe_arg.hir_id);
        if let Some(input) = fn_sig.inputs().get(i);
        let (input, n_refs) = peel_mid_ty_refs(*input);
        if let (trait_predicates, _) = get_input_traits_and_projections(cx, callee_def_id, input);
        if let Some(sized_def_id) = cx.tcx.lang_items().sized_trait();
        if let [trait_predicate] = trait_predicates
            .iter()
            .filter(|trait_predicate| trait_predicate.def_id() != sized_def_id)
            .collect::<Vec<_>>()[..];
        if let Some(deref_trait_id) = cx.tcx.get_diagnostic_item(sym::Deref);
        if let Some(as_ref_trait_id) = cx.tcx.get_diagnostic_item(sym::AsRef);
        if trait_predicate.def_id() == deref_trait_id || trait_predicate.def_id() == as_ref_trait_id;
        let receiver_ty = cx.typeck_results().expr_ty(receiver);
        // We can't add an `&` when the trait is `Deref` because `Target = &T` won't match
        // `Target = T`.
        if let Some((n_refs, receiver_ty)) = if n_refs > 0 || is_copy(cx, receiver_ty) {
            Some((n_refs, receiver_ty))
        } else if trait_predicate.def_id() != deref_trait_id {
            Some((1, Ty::new_ref(cx.tcx,
                cx.tcx.lifetimes.re_erased,
                ty::TypeAndMut {
                    ty: receiver_ty,
                    mutbl: Mutability::Not,
                },
            )))
        } else {
            None
        };
        if can_change_type(cx, maybe_arg, receiver_ty);
        if let Some(receiver_snippet) = snippet_opt(cx, receiver.span);
        then {
            span_lint_and_sugg(
                cx,
                UNNECESSARY_TO_OWNED,
                maybe_arg.span,
                &format!("unnecessary use of `{method_name}`"),
                "use",
                format!("{:&>n_refs$}{receiver_snippet}", ""),
                Applicability::MachineApplicable,
            );
            return true;
        }
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
/// `Substs`, and arguments.
fn get_callee_substs_and_args<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
) -> Option<(DefId, SubstsRef<'tcx>, Option<&'tcx Expr<'tcx>>, &'tcx [Expr<'tcx>])> {
    if_chain! {
        if let ExprKind::Call(callee, args) = expr.kind;
        let callee_ty = cx.typeck_results().expr_ty(callee);
        if let ty::FnDef(callee_def_id, _) = callee_ty.kind();
        then {
            let substs = cx.typeck_results().node_substs(callee.hir_id);
            return Some((*callee_def_id, substs, None, args));
        }
    }
    if_chain! {
        if let ExprKind::MethodCall(_, recv, args, _) = expr.kind;
        if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        then {
            let substs = cx.typeck_results().node_substs(expr.hir_id);
            return Some((method_def_id, substs, Some(recv), args));
        }
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
                if projection_predicate.projection_ty.self_ty() == input {
                    projection_predicates.push(projection_predicate);
                }
            },
            _ => {},
        }
    }
    (trait_predicates, projection_predicates)
}

fn can_change_type<'a>(cx: &LateContext<'a>, mut expr: &'a Expr<'a>, mut ty: Ty<'a>) -> bool {
    for (_, node) in cx.tcx.hir().parent_iter(expr.hir_id) {
        match node {
            Node::Stmt(_) => return true,
            Node::Block(..) => continue,
            Node::Item(item) => {
                if let ItemKind::Fn(_, _, body_id) = &item.kind
                && let output_ty = return_ty(cx, item.owner_id)
                && let inherited = Inherited::new(cx.tcx, item.owner_id.def_id)
                && let fn_ctxt = FnCtxt::new(&inherited, cx.param_env, item.owner_id.def_id)
                && fn_ctxt.can_coerce(ty, output_ty)
                {
                    if has_lifetime(output_ty) && has_lifetime(ty) {
                        return false;
                    }
                    let body = cx.tcx.hir().body(*body_id);
                    let body_expr = &body.value;
                    let mut count = 0;
                    return find_all_ret_expressions(cx, body_expr, |_| { count += 1; count <= 1 });
                }
            }
            Node::Expr(parent_expr) => {
                if let Some((callee_def_id, call_substs, recv, call_args)) = get_callee_substs_and_args(cx, parent_expr)
                {
                    // FIXME: the `subst_identity()` below seems incorrect, since we eventually
                    // call `tcx.try_subst_and_normalize_erasing_regions` further down
                    // (i.e., we are explicitly not in the identity context).
                    let fn_sig = cx.tcx.fn_sig(callee_def_id).subst_identity().skip_binder();
                    if let Some(arg_index) = recv.into_iter().chain(call_args).position(|arg| arg.hir_id == expr.hir_id)
                        && let Some(param_ty) = fn_sig.inputs().get(arg_index)
                        && let ty::Param(ParamTy { index: param_index , ..}) = param_ty.kind()
                        // https://github.com/rust-lang/rust-clippy/issues/9504 and https://github.com/rust-lang/rust-clippy/issues/10021
                        && (*param_index as usize) < call_substs.len()
                    {
                        if fn_sig
                            .inputs()
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != arg_index)
                            .any(|(_, ty)| ty.contains(*param_ty))
                        {
                            return false;
                        }

                        let mut trait_predicates = cx.tcx.param_env(callee_def_id)
                            .caller_bounds().iter().filter(|predicate| {
                            if let ClauseKind::Trait(trait_predicate)
                                    = predicate.kind().skip_binder()
                                && trait_predicate.trait_ref.self_ty() == *param_ty
                            {
                                true
                            } else {
                                false
                            }
                        });

                        let new_subst = cx.tcx.mk_substs_from_iter(
                            call_substs.iter()
                                .enumerate()
                                .map(|(i, t)|
                                     if i == (*param_index as usize) {
                                         GenericArg::from(ty)
                                     } else {
                                         t
                                     }));

                        if trait_predicates.any(|predicate| {
                            let predicate = EarlyBinder::bind(predicate).subst(cx.tcx, new_subst);
                            let obligation = Obligation::new(cx.tcx, ObligationCause::dummy(), cx.param_env, predicate);
                            !cx.tcx.infer_ctxt().build().predicate_must_hold_modulo_regions(&obligation)
                        }) {
                            return false;
                        }

                        let output_ty = fn_sig.output();
                        if output_ty.contains(*param_ty) {
                            if let Ok(new_ty)  = cx.tcx.try_subst_and_normalize_erasing_regions(
                                new_subst, cx.param_env, EarlyBinder::bind(output_ty)) {
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
    ty.walk().any(|t| matches!(t.unpack(), GenericArgKind::Lifetime(_)))
}

/// Returns true if the named method is `Iterator::cloned` or `Iterator::copied`.
fn is_cloned_or_copied(cx: &LateContext<'_>, method_name: Symbol, method_def_id: DefId) -> bool {
    (method_name.as_str() == "cloned" || method_name.as_str() == "copied")
        && is_diag_trait_item(cx, method_def_id, sym::Iterator)
}

/// Returns true if the named method can be used to convert the receiver to its "owned"
/// representation.
fn is_to_owned_like<'a>(cx: &LateContext<'a>, call_expr: &Expr<'a>, method_name: Symbol, method_def_id: DefId) -> bool {
    is_clone_like(cx, method_name.as_str(), method_def_id)
        || is_cow_into_owned(cx, method_name, method_def_id)
        || is_to_string_on_string_like(cx, call_expr, method_name, method_def_id)
}

/// Returns true if the named method is `Cow::into_owned`.
fn is_cow_into_owned(cx: &LateContext<'_>, method_name: Symbol, method_def_id: DefId) -> bool {
    method_name.as_str() == "into_owned" && is_diag_item_method(cx, method_def_id, sym::Cow)
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

    if let Some(substs) = cx.typeck_results().node_substs_opt(call_expr.hir_id)
        && let [generic_arg] = substs.as_slice()
        && let GenericArgKind::Type(ty) = generic_arg.unpack()
        && let Some(deref_trait_id) = cx.tcx.get_diagnostic_item(sym::Deref)
        && let Some(as_ref_trait_id) = cx.tcx.get_diagnostic_item(sym::AsRef)
        && (cx.get_associated_type(ty, deref_trait_id, "Target") == Some(cx.tcx.types.str_) ||
            implements_trait(cx, ty, as_ref_trait_id, &[cx.tcx.types.str_.into()])) {
            true
        } else {
            false
        }
}
