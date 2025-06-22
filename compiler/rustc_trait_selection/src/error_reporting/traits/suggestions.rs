// ignore-tidy-filelength

use std::assert_matches::debug_assert_matches;
use std::borrow::Cow;
use std::iter;

use itertools::{EitherOrBoth, Itertools};
use rustc_abi::ExternAbi;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, EmissionGuarantee, MultiSpan, Style, SuggestionStyle, pluralize,
    struct_span_code_err,
};
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{Visitor, VisitorExt};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{
    self as hir, AmbigArg, CoroutineDesugaring, CoroutineKind, CoroutineSource, Expr, HirId, Node,
    expr_needs_parens, is_range_literal,
};
use rustc_infer::infer::{BoundRegionConversionTime, DefineOpaqueTypes, InferCtxt, InferOk};
use rustc_middle::traits::IsConstable;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::print::{
    PrintPolyTraitPredicateExt as _, PrintPolyTraitRefExt, PrintTraitPredicateExt as _,
    with_forced_trimmed_paths, with_no_trimmed_paths, with_types_for_suggestion,
};
use rustc_middle::ty::{
    self, AdtKind, GenericArgs, InferTy, IsSuggestable, Ty, TyCtxt, TypeFoldable, TypeFolder,
    TypeSuperFoldable, TypeVisitableExt, TypeckResults, Upcast, suggest_arbitrary_trait_bound,
    suggest_constraining_type_param,
};
use rustc_middle::{bug, span_bug};
use rustc_span::def_id::LocalDefId;
use rustc_span::{
    BytePos, DUMMY_SP, DesugaringKind, ExpnKind, Ident, MacroKind, Span, Symbol, kw, sym,
};
use tracing::{debug, instrument};

use super::{
    DefIdOrName, FindExprBySpan, ImplCandidate, Obligation, ObligationCause, ObligationCauseCode,
    PredicateObligation,
};
use crate::error_reporting::TypeErrCtxt;
use crate::errors;
use crate::infer::InferCtxtExt as _;
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use crate::traits::{ImplDerivedCause, NormalizeExt, ObligationCtxt};

#[derive(Debug)]
pub enum CoroutineInteriorOrUpvar {
    // span of interior type
    Interior(Span, Option<(Span, Option<Span>)>),
    // span of upvar
    Upvar(Span),
}

// This type provides a uniform interface to retrieve data on coroutines, whether it originated from
// the local crate being compiled or from a foreign crate.
#[derive(Debug)]
struct CoroutineData<'a, 'tcx>(&'a TypeckResults<'tcx>);

impl<'a, 'tcx> CoroutineData<'a, 'tcx> {
    /// Try to get information about variables captured by the coroutine that matches a type we are
    /// looking for with `ty_matches` function. We uses it to find upvar which causes a failure to
    /// meet an obligation
    fn try_get_upvar_span<F>(
        &self,
        infer_context: &InferCtxt<'tcx>,
        coroutine_did: DefId,
        ty_matches: F,
    ) -> Option<CoroutineInteriorOrUpvar>
    where
        F: Fn(ty::Binder<'tcx, Ty<'tcx>>) -> bool,
    {
        infer_context.tcx.upvars_mentioned(coroutine_did).and_then(|upvars| {
            upvars.iter().find_map(|(upvar_id, upvar)| {
                let upvar_ty = self.0.node_type(*upvar_id);
                let upvar_ty = infer_context.resolve_vars_if_possible(upvar_ty);
                ty_matches(ty::Binder::dummy(upvar_ty))
                    .then(|| CoroutineInteriorOrUpvar::Upvar(upvar.span))
            })
        })
    }

    /// Try to get the span of a type being awaited on that matches the type we are looking with the
    /// `ty_matches` function. We uses it to find awaited type which causes a failure to meet an
    /// obligation
    fn get_from_await_ty<F>(
        &self,
        visitor: AwaitsVisitor,
        tcx: TyCtxt<'tcx>,
        ty_matches: F,
    ) -> Option<Span>
    where
        F: Fn(ty::Binder<'tcx, Ty<'tcx>>) -> bool,
    {
        visitor
            .awaits
            .into_iter()
            .map(|id| tcx.hir_expect_expr(id))
            .find(|await_expr| ty_matches(ty::Binder::dummy(self.0.expr_ty_adjusted(await_expr))))
            .map(|expr| expr.span)
    }
}

fn predicate_constraint(generics: &hir::Generics<'_>, pred: ty::Predicate<'_>) -> (Span, String) {
    (
        generics.tail_span_for_predicate_suggestion(),
        with_types_for_suggestion!(format!("{} {}", generics.add_where_or_trailing_comma(), pred)),
    )
}

/// Type parameter needs more bounds. The trivial case is `T` `where T: Bound`, but
/// it can also be an `impl Trait` param that needs to be decomposed to a type
/// param for cleaner code.
pub fn suggest_restriction<'tcx, G: EmissionGuarantee>(
    tcx: TyCtxt<'tcx>,
    item_id: LocalDefId,
    hir_generics: &hir::Generics<'tcx>,
    msg: &str,
    err: &mut Diag<'_, G>,
    fn_sig: Option<&hir::FnSig<'_>>,
    projection: Option<ty::AliasTy<'_>>,
    trait_pred: ty::PolyTraitPredicate<'tcx>,
    // When we are dealing with a trait, `super_traits` will be `Some`:
    // Given `trait T: A + B + C {}`
    //              -  ^^^^^^^^^ GenericBounds
    //              |
    //              &Ident
    super_traits: Option<(&Ident, &hir::GenericBounds<'_>)>,
) {
    if hir_generics.where_clause_span.from_expansion()
        || hir_generics.where_clause_span.desugaring_kind().is_some()
        || projection.is_some_and(|projection| {
            (tcx.is_impl_trait_in_trait(projection.def_id)
                && !tcx.features().return_type_notation())
                || tcx.lookup_stability(projection.def_id).is_some_and(|stab| stab.is_unstable())
        })
    {
        return;
    }
    let generics = tcx.generics_of(item_id);
    // Given `fn foo(t: impl Trait)` where `Trait` requires assoc type `A`...
    if let Some((param, bound_str, fn_sig)) =
        fn_sig.zip(projection).and_then(|(sig, p)| match *p.self_ty().kind() {
            // Shenanigans to get the `Trait` from the `impl Trait`.
            ty::Param(param) => {
                let param_def = generics.type_param(param, tcx);
                if param_def.kind.is_synthetic() {
                    let bound_str =
                        param_def.name.as_str().strip_prefix("impl ")?.trim_start().to_string();
                    return Some((param_def, bound_str, sig));
                }
                None
            }
            _ => None,
        })
    {
        let type_param_name = hir_generics.params.next_type_param_name(Some(&bound_str));
        let trait_pred = trait_pred.fold_with(&mut ReplaceImplTraitFolder {
            tcx,
            param,
            replace_ty: ty::ParamTy::new(generics.count() as u32, Symbol::intern(&type_param_name))
                .to_ty(tcx),
        });
        if !trait_pred.is_suggestable(tcx, false) {
            return;
        }
        // We know we have an `impl Trait` that doesn't satisfy a required projection.

        // Find all of the occurrences of `impl Trait` for `Trait` in the function arguments'
        // types. There should be at least one, but there might be *more* than one. In that
        // case we could just ignore it and try to identify which one needs the restriction,
        // but instead we choose to suggest replacing all instances of `impl Trait` with `T`
        // where `T: Trait`.
        let mut ty_spans = vec![];
        for input in fn_sig.decl.inputs {
            ReplaceImplTraitVisitor { ty_spans: &mut ty_spans, param_did: param.def_id }
                .visit_ty_unambig(input);
        }
        // The type param `T: Trait` we will suggest to introduce.
        let type_param = format!("{type_param_name}: {bound_str}");

        let mut sugg = vec![
            if let Some(span) = hir_generics.span_for_param_suggestion() {
                (span, format!(", {type_param}"))
            } else {
                (hir_generics.span, format!("<{type_param}>"))
            },
            // `fn foo(t: impl Trait)`
            //                       ^ suggest `where <T as Trait>::A: Bound`
            predicate_constraint(hir_generics, trait_pred.upcast(tcx)),
        ];
        sugg.extend(ty_spans.into_iter().map(|s| (s, type_param_name.to_string())));

        // Suggest `fn foo<T: Trait>(t: T) where <T as Trait>::A: Bound`.
        // FIXME: we should suggest `fn foo(t: impl Trait<A: Bound>)` instead.
        err.multipart_suggestion(
            "introduce a type parameter with a trait bound instead of using `impl Trait`",
            sugg,
            Applicability::MaybeIncorrect,
        );
    } else {
        if !trait_pred.is_suggestable(tcx, false) {
            return;
        }
        // Trivial case: `T` needs an extra bound: `T: Bound`.
        let (sp, suggestion) = match (
            hir_generics
                .params
                .iter()
                .find(|p| !matches!(p.kind, hir::GenericParamKind::Type { synthetic: true, .. })),
            super_traits,
        ) {
            (_, None) => predicate_constraint(hir_generics, trait_pred.upcast(tcx)),
            (None, Some((ident, []))) => (
                ident.span.shrink_to_hi(),
                format!(": {}", trait_pred.print_modifiers_and_trait_path()),
            ),
            (_, Some((_, [.., bounds]))) => (
                bounds.span().shrink_to_hi(),
                format!(" + {}", trait_pred.print_modifiers_and_trait_path()),
            ),
            (Some(_), Some((_, []))) => (
                hir_generics.span.shrink_to_hi(),
                format!(": {}", trait_pred.print_modifiers_and_trait_path()),
            ),
        };

        err.span_suggestion_verbose(
            sp,
            format!("consider further restricting {msg}"),
            suggestion,
            Applicability::MachineApplicable,
        );
    }
}

impl<'a, 'tcx> TypeErrCtxt<'a, 'tcx> {
    pub fn suggest_restricting_param_bound(
        &self,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        associated_ty: Option<(&'static str, Ty<'tcx>)>,
        mut body_id: LocalDefId,
    ) {
        if trait_pred.skip_binder().polarity != ty::PredicatePolarity::Positive {
            return;
        }

        let trait_pred = self.resolve_numeric_literals_with_default(trait_pred);

        let self_ty = trait_pred.skip_binder().self_ty();
        let (param_ty, projection) = match *self_ty.kind() {
            ty::Param(_) => (true, None),
            ty::Alias(ty::Projection, projection) => (false, Some(projection)),
            _ => (false, None),
        };

        // FIXME: Add check for trait bound that is already present, particularly `?Sized` so we
        //        don't suggest `T: Sized + ?Sized`.
        loop {
            let node = self.tcx.hir_node_by_def_id(body_id);
            match node {
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Trait(_, _, ident, generics, bounds, _),
                    ..
                }) if self_ty == self.tcx.types.self_param => {
                    assert!(param_ty);
                    // Restricting `Self` for a single method.
                    suggest_restriction(
                        self.tcx,
                        body_id,
                        generics,
                        "`Self`",
                        err,
                        None,
                        projection,
                        trait_pred,
                        Some((&ident, bounds)),
                    );
                    return;
                }

                hir::Node::TraitItem(hir::TraitItem {
                    generics,
                    kind: hir::TraitItemKind::Fn(..),
                    ..
                }) if self_ty == self.tcx.types.self_param => {
                    assert!(param_ty);
                    // Restricting `Self` for a single method.
                    suggest_restriction(
                        self.tcx, body_id, generics, "`Self`", err, None, projection, trait_pred,
                        None,
                    );
                    return;
                }

                hir::Node::TraitItem(hir::TraitItem {
                    generics,
                    kind: hir::TraitItemKind::Fn(fn_sig, ..),
                    ..
                })
                | hir::Node::ImplItem(hir::ImplItem {
                    generics,
                    kind: hir::ImplItemKind::Fn(fn_sig, ..),
                    ..
                })
                | hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Fn { sig: fn_sig, generics, .. },
                    ..
                }) if projection.is_some() => {
                    // Missing restriction on associated type of type parameter (unmet projection).
                    suggest_restriction(
                        self.tcx,
                        body_id,
                        generics,
                        "the associated type",
                        err,
                        Some(fn_sig),
                        projection,
                        trait_pred,
                        None,
                    );
                    return;
                }
                hir::Node::Item(hir::Item {
                    kind:
                        hir::ItemKind::Trait(_, _, _, generics, ..)
                        | hir::ItemKind::Impl(hir::Impl { generics, .. }),
                    ..
                }) if projection.is_some() => {
                    // Missing restriction on associated type of type parameter (unmet projection).
                    suggest_restriction(
                        self.tcx,
                        body_id,
                        generics,
                        "the associated type",
                        err,
                        None,
                        projection,
                        trait_pred,
                        None,
                    );
                    return;
                }

                hir::Node::Item(hir::Item {
                    kind:
                        hir::ItemKind::Struct(_, generics, _)
                        | hir::ItemKind::Enum(_, generics, _)
                        | hir::ItemKind::Union(_, generics, _)
                        | hir::ItemKind::Trait(_, _, _, generics, ..)
                        | hir::ItemKind::Impl(hir::Impl { generics, .. })
                        | hir::ItemKind::Fn { generics, .. }
                        | hir::ItemKind::TyAlias(_, generics, _)
                        | hir::ItemKind::Const(_, generics, _, _)
                        | hir::ItemKind::TraitAlias(_, generics, _),
                    ..
                })
                | hir::Node::TraitItem(hir::TraitItem { generics, .. })
                | hir::Node::ImplItem(hir::ImplItem { generics, .. })
                    if param_ty =>
                {
                    // We skip the 0'th arg (self) because we do not want
                    // to consider the predicate as not suggestible if the
                    // self type is an arg position `impl Trait` -- instead,
                    // we handle that by adding ` + Bound` below.
                    // FIXME(compiler-errors): It would be nice to do the same
                    // this that we do in `suggest_restriction` and pull the
                    // `impl Trait` into a new generic if it shows up somewhere
                    // else in the predicate.
                    if !trait_pred.skip_binder().trait_ref.args[1..]
                        .iter()
                        .all(|g| g.is_suggestable(self.tcx, false))
                    {
                        return;
                    }
                    // Missing generic type parameter bound.
                    let param_name = self_ty.to_string();
                    let mut constraint = with_no_trimmed_paths!(
                        trait_pred.print_modifiers_and_trait_path().to_string()
                    );

                    if let Some((name, term)) = associated_ty {
                        // FIXME: this case overlaps with code in TyCtxt::note_and_explain_type_err.
                        // That should be extracted into a helper function.
                        if let Some(stripped) = constraint.strip_suffix('>') {
                            constraint = format!("{stripped}, {name} = {term}>");
                        } else {
                            constraint.push_str(&format!("<{name} = {term}>"));
                        }
                    }

                    if suggest_constraining_type_param(
                        self.tcx,
                        generics,
                        err,
                        &param_name,
                        &constraint,
                        Some(trait_pred.def_id()),
                        None,
                    ) {
                        return;
                    }
                }

                hir::Node::Item(hir::Item {
                    kind:
                        hir::ItemKind::Struct(_, generics, _)
                        | hir::ItemKind::Enum(_, generics, _)
                        | hir::ItemKind::Union(_, generics, _)
                        | hir::ItemKind::Trait(_, _, _, generics, ..)
                        | hir::ItemKind::Impl(hir::Impl { generics, .. })
                        | hir::ItemKind::Fn { generics, .. }
                        | hir::ItemKind::TyAlias(_, generics, _)
                        | hir::ItemKind::Const(_, generics, _, _)
                        | hir::ItemKind::TraitAlias(_, generics, _),
                    ..
                }) if !param_ty => {
                    // Missing generic type parameter bound.
                    if suggest_arbitrary_trait_bound(
                        self.tcx,
                        generics,
                        err,
                        trait_pred,
                        associated_ty,
                    ) {
                        return;
                    }
                }
                hir::Node::Crate(..) => return,

                _ => {}
            }
            body_id = self.tcx.local_parent(body_id);
        }
    }

    /// Provide a suggestion to dereference arguments to functions and binary operators, if that
    /// would satisfy trait bounds.
    pub(super) fn suggest_dereferences(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        let mut code = obligation.cause.code();
        if let ObligationCauseCode::FunctionArg { arg_hir_id, call_hir_id, .. } = code
            && let Some(typeck_results) = &self.typeck_results
            && let hir::Node::Expr(expr) = self.tcx.hir_node(*arg_hir_id)
            && let Some(arg_ty) = typeck_results.expr_ty_adjusted_opt(expr)
        {
            // Suggest dereferencing the argument to a function/method call if possible

            // Get the root obligation, since the leaf obligation we have may be unhelpful (#87437)
            let mut real_trait_pred = trait_pred;
            while let Some((parent_code, parent_trait_pred)) = code.parent_with_predicate() {
                code = parent_code;
                if let Some(parent_trait_pred) = parent_trait_pred {
                    real_trait_pred = parent_trait_pred;
                }
            }

            // We `instantiate_bound_regions_with_erased` here because `make_subregion` does not handle
            // `ReBound`, and we don't particularly care about the regions.
            let real_ty = self.tcx.instantiate_bound_regions_with_erased(real_trait_pred.self_ty());
            if !self.can_eq(obligation.param_env, real_ty, arg_ty) {
                return false;
            }

            // Potentially, we'll want to place our dereferences under a `&`. We don't try this for
            // `&mut`, since we can't be sure users will get the side-effects they want from it.
            // If this doesn't work, we'll try removing the `&` in `suggest_remove_reference`.
            // FIXME(dianne): this misses the case where users need both to deref and remove `&`s.
            // This method could be combined with `TypeErrCtxt::suggest_remove_reference` to handle
            // that, similar to what `FnCtxt::suggest_deref_or_ref` does.
            let (is_under_ref, base_ty, span) = match expr.kind {
                hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, subexpr)
                    if let &ty::Ref(region, base_ty, hir::Mutability::Not) = real_ty.kind() =>
                {
                    (Some(region), base_ty, subexpr.span)
                }
                // Don't suggest `*&mut`, etc.
                hir::ExprKind::AddrOf(..) => return false,
                _ => (None, real_ty, obligation.cause.span),
            };

            let autoderef = (self.autoderef_steps)(base_ty);
            let mut is_boxed = base_ty.is_box();
            if let Some(steps) = autoderef.into_iter().position(|(mut ty, obligations)| {
                // Ensure one of the following for dereferencing to be valid: we're passing by
                // reference, `ty` is `Copy`, or we're moving out of a (potentially nested) `Box`.
                let can_deref = is_under_ref.is_some()
                    || self.type_is_copy_modulo_regions(obligation.param_env, ty)
                    || ty.is_numeric() // for inference vars (presumably but not provably `Copy`)
                    || is_boxed && self.type_is_sized_modulo_regions(obligation.param_env, ty);
                is_boxed &= ty.is_box();

                // Re-add the `&` if necessary
                if let Some(region) = is_under_ref {
                    ty = Ty::new_ref(self.tcx, region, ty, hir::Mutability::Not);
                }

                // Remapping bound vars here
                let real_trait_pred_and_ty =
                    real_trait_pred.map_bound(|inner_trait_pred| (inner_trait_pred, ty));
                let obligation = self.mk_trait_obligation_with_new_self_ty(
                    obligation.param_env,
                    real_trait_pred_and_ty,
                );

                can_deref
                    && obligations
                        .iter()
                        .chain([&obligation])
                        .all(|obligation| self.predicate_may_hold(obligation))
            }) && steps > 0
            {
                let derefs = "*".repeat(steps);
                let msg = "consider dereferencing here";
                let call_node = self.tcx.hir_node(*call_hir_id);
                let is_receiver = matches!(
                    call_node,
                    Node::Expr(hir::Expr {
                        kind: hir::ExprKind::MethodCall(_, receiver_expr, ..),
                        ..
                    })
                    if receiver_expr.hir_id == *arg_hir_id
                );
                if is_receiver {
                    err.multipart_suggestion_verbose(
                        msg,
                        vec![
                            (span.shrink_to_lo(), format!("({derefs}")),
                            (span.shrink_to_hi(), ")".to_string()),
                        ],
                        Applicability::MachineApplicable,
                    )
                } else {
                    err.span_suggestion_verbose(
                        span.shrink_to_lo(),
                        msg,
                        derefs,
                        Applicability::MachineApplicable,
                    )
                };
                return true;
            }
        } else if let (
            ObligationCauseCode::BinOp { lhs_hir_id, rhs_hir_id: Some(rhs_hir_id), .. },
            predicate,
        ) = code.peel_derives_with_predicate()
            && let Some(typeck_results) = &self.typeck_results
            && let hir::Node::Expr(lhs) = self.tcx.hir_node(*lhs_hir_id)
            && let hir::Node::Expr(rhs) = self.tcx.hir_node(*rhs_hir_id)
            && let Some(rhs_ty) = typeck_results.expr_ty_opt(rhs)
            && let trait_pred = predicate.unwrap_or(trait_pred)
            // Only run this code on binary operators
            && hir::lang_items::BINARY_OPERATORS
                .iter()
                .filter_map(|&op| self.tcx.lang_items().get(op))
                .any(|op| {
                    op == trait_pred.skip_binder().trait_ref.def_id
                })
        {
            // Suggest dereferencing the LHS, RHS, or both terms of a binop if possible

            let trait_pred = predicate.unwrap_or(trait_pred);
            let lhs_ty = self.tcx.instantiate_bound_regions_with_erased(trait_pred.self_ty());
            let lhs_autoderef = (self.autoderef_steps)(lhs_ty);
            let rhs_autoderef = (self.autoderef_steps)(rhs_ty);
            let first_lhs = lhs_autoderef.first().unwrap().clone();
            let first_rhs = rhs_autoderef.first().unwrap().clone();
            let mut autoderefs = lhs_autoderef
                .into_iter()
                .enumerate()
                .rev()
                .zip_longest(rhs_autoderef.into_iter().enumerate().rev())
                .map(|t| match t {
                    EitherOrBoth::Both(a, b) => (a, b),
                    EitherOrBoth::Left(a) => (a, (0, first_rhs.clone())),
                    EitherOrBoth::Right(b) => ((0, first_lhs.clone()), b),
                })
                .rev();
            if let Some((lsteps, rsteps)) =
                autoderefs.find_map(|((lsteps, (l_ty, _)), (rsteps, (r_ty, _)))| {
                    // Create a new predicate with the dereferenced LHS and RHS
                    // We simultaneously dereference both sides rather than doing them
                    // one at a time to account for cases such as &Box<T> == &&T
                    let trait_pred_and_ty = trait_pred.map_bound(|inner| {
                        (
                            ty::TraitPredicate {
                                trait_ref: ty::TraitRef::new_from_args(
                                    self.tcx,
                                    inner.trait_ref.def_id,
                                    self.tcx.mk_args(
                                        &[&[l_ty.into(), r_ty.into()], &inner.trait_ref.args[2..]]
                                            .concat(),
                                    ),
                                ),
                                ..inner
                            },
                            l_ty,
                        )
                    });
                    let obligation = self.mk_trait_obligation_with_new_self_ty(
                        obligation.param_env,
                        trait_pred_and_ty,
                    );
                    self.predicate_may_hold(&obligation).then_some(match (lsteps, rsteps) {
                        (_, 0) => (Some(lsteps), None),
                        (0, _) => (None, Some(rsteps)),
                        _ => (Some(lsteps), Some(rsteps)),
                    })
                })
            {
                let make_sugg = |mut expr: &Expr<'_>, mut steps| {
                    let mut prefix_span = expr.span.shrink_to_lo();
                    let mut msg = "consider dereferencing here";
                    if let hir::ExprKind::AddrOf(_, _, inner) = expr.kind {
                        msg = "consider removing the borrow and dereferencing instead";
                        if let hir::ExprKind::AddrOf(..) = inner.kind {
                            msg = "consider removing the borrows and dereferencing instead";
                        }
                    }
                    while let hir::ExprKind::AddrOf(_, _, inner) = expr.kind
                        && steps > 0
                    {
                        prefix_span = prefix_span.with_hi(inner.span.lo());
                        expr = inner;
                        steps -= 1;
                    }
                    // Empty suggestions with empty spans ICE with debug assertions
                    if steps == 0 {
                        return (
                            msg.trim_end_matches(" and dereferencing instead"),
                            vec![(prefix_span, String::new())],
                        );
                    }
                    let derefs = "*".repeat(steps);
                    let needs_parens = steps > 0
                        && match expr.kind {
                            hir::ExprKind::Cast(_, _) | hir::ExprKind::Binary(_, _, _) => true,
                            _ if is_range_literal(expr) => true,
                            _ => false,
                        };
                    let mut suggestion = if needs_parens {
                        vec![
                            (
                                expr.span.with_lo(prefix_span.hi()).shrink_to_lo(),
                                format!("{derefs}("),
                            ),
                            (expr.span.shrink_to_hi(), ")".to_string()),
                        ]
                    } else {
                        vec![(
                            expr.span.with_lo(prefix_span.hi()).shrink_to_lo(),
                            format!("{derefs}"),
                        )]
                    };
                    // Empty suggestions with empty spans ICE with debug assertions
                    if !prefix_span.is_empty() {
                        suggestion.push((prefix_span, String::new()));
                    }
                    (msg, suggestion)
                };

                if let Some(lsteps) = lsteps
                    && let Some(rsteps) = rsteps
                    && lsteps > 0
                    && rsteps > 0
                {
                    let mut suggestion = make_sugg(lhs, lsteps).1;
                    suggestion.append(&mut make_sugg(rhs, rsteps).1);
                    err.multipart_suggestion_verbose(
                        "consider dereferencing both sides of the expression",
                        suggestion,
                        Applicability::MachineApplicable,
                    );
                    return true;
                } else if let Some(lsteps) = lsteps
                    && lsteps > 0
                {
                    let (msg, suggestion) = make_sugg(lhs, lsteps);
                    err.multipart_suggestion_verbose(
                        msg,
                        suggestion,
                        Applicability::MachineApplicable,
                    );
                    return true;
                } else if let Some(rsteps) = rsteps
                    && rsteps > 0
                {
                    let (msg, suggestion) = make_sugg(rhs, rsteps);
                    err.multipart_suggestion_verbose(
                        msg,
                        suggestion,
                        Applicability::MachineApplicable,
                    );
                    return true;
                }
            }
        }
        false
    }

    /// Given a closure's `DefId`, return the given name of the closure.
    ///
    /// This doesn't account for reassignments, but it's only used for suggestions.
    fn get_closure_name(
        &self,
        def_id: DefId,
        err: &mut Diag<'_>,
        msg: Cow<'static, str>,
    ) -> Option<Symbol> {
        let get_name = |err: &mut Diag<'_>, kind: &hir::PatKind<'_>| -> Option<Symbol> {
            // Get the local name of this closure. This can be inaccurate because
            // of the possibility of reassignment, but this should be good enough.
            match &kind {
                hir::PatKind::Binding(hir::BindingMode::NONE, _, ident, None) => Some(ident.name),
                _ => {
                    err.note(msg);
                    None
                }
            }
        };

        let hir_id = self.tcx.local_def_id_to_hir_id(def_id.as_local()?);
        match self.tcx.parent_hir_node(hir_id) {
            hir::Node::Stmt(hir::Stmt { kind: hir::StmtKind::Let(local), .. }) => {
                get_name(err, &local.pat.kind)
            }
            // Different to previous arm because one is `&hir::Local` and the other
            // is `P<hir::Local>`.
            hir::Node::LetStmt(local) => get_name(err, &local.pat.kind),
            _ => None,
        }
    }

    /// We tried to apply the bound to an `fn` or closure. Check whether calling it would
    /// evaluate to a type that *would* satisfy the trait bound. If it would, suggest calling
    /// it: `bar(foo)` → `bar(foo())`. This case is *very* likely to be hit if `foo` is `async`.
    pub(super) fn suggest_fn_call(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        // It doesn't make sense to make this suggestion outside of typeck...
        // (also autoderef will ICE...)
        if self.typeck_results.is_none() {
            return false;
        }

        if let ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_pred)) =
            obligation.predicate.kind().skip_binder()
            && self.tcx.is_lang_item(trait_pred.def_id(), LangItem::Sized)
        {
            // Don't suggest calling to turn an unsized type into a sized type
            return false;
        }

        let self_ty = self.instantiate_binder_with_fresh_vars(
            DUMMY_SP,
            BoundRegionConversionTime::FnCall,
            trait_pred.self_ty(),
        );

        let Some((def_id_or_name, output, inputs)) =
            self.extract_callable_info(obligation.cause.body_id, obligation.param_env, self_ty)
        else {
            return false;
        };

        // Remapping bound vars here
        let trait_pred_and_self = trait_pred.map_bound(|trait_pred| (trait_pred, output));

        let new_obligation =
            self.mk_trait_obligation_with_new_self_ty(obligation.param_env, trait_pred_and_self);
        if !self.predicate_must_hold_modulo_regions(&new_obligation) {
            return false;
        }

        // Get the name of the callable and the arguments to be used in the suggestion.
        let msg = match def_id_or_name {
            DefIdOrName::DefId(def_id) => match self.tcx.def_kind(def_id) {
                DefKind::Ctor(CtorOf::Struct, _) => {
                    Cow::from("use parentheses to construct this tuple struct")
                }
                DefKind::Ctor(CtorOf::Variant, _) => {
                    Cow::from("use parentheses to construct this tuple variant")
                }
                kind => Cow::from(format!(
                    "use parentheses to call this {}",
                    self.tcx.def_kind_descr(kind, def_id)
                )),
            },
            DefIdOrName::Name(name) => Cow::from(format!("use parentheses to call this {name}")),
        };

        let args = inputs
            .into_iter()
            .map(|ty| {
                if ty.is_suggestable(self.tcx, false) {
                    format!("/* {ty} */")
                } else {
                    "/* value */".to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        if matches!(obligation.cause.code(), ObligationCauseCode::FunctionArg { .. })
            && obligation.cause.span.can_be_used_for_suggestions()
        {
            // When the obligation error has been ensured to have been caused by
            // an argument, the `obligation.cause.span` points at the expression
            // of the argument, so we can provide a suggestion. Otherwise, we give
            // a more general note.
            err.span_suggestion_verbose(
                obligation.cause.span.shrink_to_hi(),
                msg,
                format!("({args})"),
                Applicability::HasPlaceholders,
            );
        } else if let DefIdOrName::DefId(def_id) = def_id_or_name {
            let name = match self.tcx.hir_get_if_local(def_id) {
                Some(hir::Node::Expr(hir::Expr {
                    kind: hir::ExprKind::Closure(hir::Closure { fn_decl_span, .. }),
                    ..
                })) => {
                    err.span_label(*fn_decl_span, "consider calling this closure");
                    let Some(name) = self.get_closure_name(def_id, err, msg.clone()) else {
                        return false;
                    };
                    name.to_string()
                }
                Some(hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Fn { ident, .. }, ..
                })) => {
                    err.span_label(ident.span, "consider calling this function");
                    ident.to_string()
                }
                Some(hir::Node::Ctor(..)) => {
                    let name = self.tcx.def_path_str(def_id);
                    err.span_label(
                        self.tcx.def_span(def_id),
                        format!("consider calling the constructor for `{name}`"),
                    );
                    name
                }
                _ => return false,
            };
            err.help(format!("{msg}: `{name}({args})`"));
        }
        true
    }

    pub(super) fn check_for_binding_assigned_block_without_tail_expression(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) {
        let mut span = obligation.cause.span;
        while span.from_expansion() {
            // Remove all the desugaring and macro contexts.
            span.remove_mark();
        }
        let mut expr_finder = FindExprBySpan::new(span, self.tcx);
        let Some(body) = self.tcx.hir_maybe_body_owned_by(obligation.cause.body_id) else {
            return;
        };
        expr_finder.visit_expr(body.value);
        let Some(expr) = expr_finder.result else {
            return;
        };
        let Some(typeck) = &self.typeck_results else {
            return;
        };
        let Some(ty) = typeck.expr_ty_adjusted_opt(expr) else {
            return;
        };
        if !ty.is_unit() {
            return;
        };
        let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = expr.kind else {
            return;
        };
        let Res::Local(hir_id) = path.res else {
            return;
        };
        let hir::Node::Pat(pat) = self.tcx.hir_node(hir_id) else {
            return;
        };
        let hir::Node::LetStmt(hir::LetStmt { ty: None, init: Some(init), .. }) =
            self.tcx.parent_hir_node(pat.hir_id)
        else {
            return;
        };
        let hir::ExprKind::Block(block, None) = init.kind else {
            return;
        };
        if block.expr.is_some() {
            return;
        }
        let [.., stmt] = block.stmts else {
            err.span_label(block.span, "this empty block is missing a tail expression");
            return;
        };
        let hir::StmtKind::Semi(tail_expr) = stmt.kind else {
            return;
        };
        let Some(ty) = typeck.expr_ty_opt(tail_expr) else {
            err.span_label(block.span, "this block is missing a tail expression");
            return;
        };
        let ty = self.resolve_numeric_literals_with_default(self.resolve_vars_if_possible(ty));
        let trait_pred_and_self = trait_pred.map_bound(|trait_pred| (trait_pred, ty));

        let new_obligation =
            self.mk_trait_obligation_with_new_self_ty(obligation.param_env, trait_pred_and_self);
        if self.predicate_must_hold_modulo_regions(&new_obligation) {
            err.span_suggestion_short(
                stmt.span.with_lo(tail_expr.span.hi()),
                "remove this semicolon",
                "",
                Applicability::MachineApplicable,
            );
        } else {
            err.span_label(block.span, "this block is missing a tail expression");
        }
    }

    pub(super) fn suggest_add_clone_to_arg(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        let self_ty = self.resolve_vars_if_possible(trait_pred.self_ty());
        self.enter_forall(self_ty, |ty: Ty<'_>| {
            let Some(generics) = self.tcx.hir_get_generics(obligation.cause.body_id) else {
                return false;
            };
            let ty::Ref(_, inner_ty, hir::Mutability::Not) = ty.kind() else { return false };
            let ty::Param(param) = inner_ty.kind() else { return false };
            let ObligationCauseCode::FunctionArg { arg_hir_id, .. } = obligation.cause.code()
            else {
                return false;
            };

            let clone_trait = self.tcx.require_lang_item(LangItem::Clone, obligation.cause.span);
            let has_clone = |ty| {
                self.type_implements_trait(clone_trait, [ty], obligation.param_env)
                    .must_apply_modulo_regions()
            };

            let existing_clone_call = match self.tcx.hir_node(*arg_hir_id) {
                // It's just a variable. Propose cloning it.
                Node::Expr(Expr { kind: hir::ExprKind::Path(_), .. }) => None,
                // It's already a call to `clone()`. We might be able to suggest
                // adding a `+ Clone` bound, though.
                Node::Expr(Expr {
                    kind:
                        hir::ExprKind::MethodCall(
                            hir::PathSegment { ident, .. },
                            _receiver,
                            [],
                            call_span,
                        ),
                    hir_id,
                    ..
                }) if ident.name == sym::clone
                    && !call_span.from_expansion()
                    && !has_clone(*inner_ty) =>
                {
                    // We only care about method calls corresponding to the real `Clone` trait.
                    let Some(typeck_results) = self.typeck_results.as_ref() else { return false };
                    let Some((DefKind::AssocFn, did)) = typeck_results.type_dependent_def(*hir_id)
                    else {
                        return false;
                    };
                    if self.tcx.trait_of_item(did) != Some(clone_trait) {
                        return false;
                    }
                    Some(ident.span)
                }
                _ => return false,
            };

            let new_obligation = self.mk_trait_obligation_with_new_self_ty(
                obligation.param_env,
                trait_pred.map_bound(|trait_pred| (trait_pred, *inner_ty)),
            );

            if self.predicate_may_hold(&new_obligation) && has_clone(ty) {
                if !has_clone(param.to_ty(self.tcx)) {
                    suggest_constraining_type_param(
                        self.tcx,
                        generics,
                        err,
                        param.name.as_str(),
                        "Clone",
                        Some(clone_trait),
                        None,
                    );
                }
                if let Some(existing_clone_call) = existing_clone_call {
                    err.span_note(
                        existing_clone_call,
                        format!(
                            "this `clone()` copies the reference, \
                            which does not do anything, \
                            because `{inner_ty}` does not implement `Clone`"
                        ),
                    );
                } else {
                    err.span_suggestion_verbose(
                        obligation.cause.span.shrink_to_hi(),
                        "consider using clone here",
                        ".clone()".to_string(),
                        Applicability::MaybeIncorrect,
                    );
                }
                return true;
            }
            false
        })
    }

    /// Extracts information about a callable type for diagnostics. This is a
    /// heuristic -- it doesn't necessarily mean that a type is always callable,
    /// because the callable type must also be well-formed to be called.
    pub fn extract_callable_info(
        &self,
        body_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        found: Ty<'tcx>,
    ) -> Option<(DefIdOrName, Ty<'tcx>, Vec<Ty<'tcx>>)> {
        // Autoderef is useful here because sometimes we box callables, etc.
        let Some((def_id_or_name, output, inputs)) =
            (self.autoderef_steps)(found).into_iter().find_map(|(found, _)| match *found.kind() {
                ty::FnPtr(sig_tys, _) => Some((
                    DefIdOrName::Name("function pointer"),
                    sig_tys.output(),
                    sig_tys.inputs(),
                )),
                ty::FnDef(def_id, _) => {
                    let fn_sig = found.fn_sig(self.tcx);
                    Some((DefIdOrName::DefId(def_id), fn_sig.output(), fn_sig.inputs()))
                }
                ty::Closure(def_id, args) => {
                    let fn_sig = args.as_closure().sig();
                    Some((
                        DefIdOrName::DefId(def_id),
                        fn_sig.output(),
                        fn_sig.inputs().map_bound(|inputs| inputs[0].tuple_fields().as_slice()),
                    ))
                }
                ty::CoroutineClosure(def_id, args) => {
                    let sig_parts = args.as_coroutine_closure().coroutine_closure_sig();
                    Some((
                        DefIdOrName::DefId(def_id),
                        sig_parts.map_bound(|sig| {
                            sig.to_coroutine(
                                self.tcx,
                                args.as_coroutine_closure().parent_args(),
                                // Just use infer vars here, since we  don't really care
                                // what these types are, just that we're returning a coroutine.
                                self.next_ty_var(DUMMY_SP),
                                self.tcx.coroutine_for_closure(def_id),
                                self.next_ty_var(DUMMY_SP),
                            )
                        }),
                        sig_parts.map_bound(|sig| sig.tupled_inputs_ty.tuple_fields().as_slice()),
                    ))
                }
                ty::Alias(ty::Opaque, ty::AliasTy { def_id, args, .. }) => {
                    self.tcx.item_self_bounds(def_id).instantiate(self.tcx, args).iter().find_map(
                        |pred| {
                            if let ty::ClauseKind::Projection(proj) = pred.kind().skip_binder()
                            && self
                                .tcx
                                .is_lang_item(proj.projection_term.def_id, LangItem::FnOnceOutput)
                            // args tuple will always be args[1]
                            && let ty::Tuple(args) = proj.projection_term.args.type_at(1).kind()
                            {
                                Some((
                                    DefIdOrName::DefId(def_id),
                                    pred.kind().rebind(proj.term.expect_type()),
                                    pred.kind().rebind(args.as_slice()),
                                ))
                            } else {
                                None
                            }
                        },
                    )
                }
                ty::Dynamic(data, _, ty::Dyn) => data.iter().find_map(|pred| {
                    if let ty::ExistentialPredicate::Projection(proj) = pred.skip_binder()
                        && self.tcx.is_lang_item(proj.def_id, LangItem::FnOnceOutput)
                        // for existential projection, args are shifted over by 1
                        && let ty::Tuple(args) = proj.args.type_at(0).kind()
                    {
                        Some((
                            DefIdOrName::Name("trait object"),
                            pred.rebind(proj.term.expect_type()),
                            pred.rebind(args.as_slice()),
                        ))
                    } else {
                        None
                    }
                }),
                ty::Param(param) => {
                    let generics = self.tcx.generics_of(body_id);
                    let name = if generics.count() > param.index as usize
                        && let def = generics.param_at(param.index as usize, self.tcx)
                        && matches!(def.kind, ty::GenericParamDefKind::Type { .. })
                        && def.name == param.name
                    {
                        DefIdOrName::DefId(def.def_id)
                    } else {
                        DefIdOrName::Name("type parameter")
                    };
                    param_env.caller_bounds().iter().find_map(|pred| {
                        if let ty::ClauseKind::Projection(proj) = pred.kind().skip_binder()
                            && self
                                .tcx
                                .is_lang_item(proj.projection_term.def_id, LangItem::FnOnceOutput)
                            && proj.projection_term.self_ty() == found
                            // args tuple will always be args[1]
                            && let ty::Tuple(args) = proj.projection_term.args.type_at(1).kind()
                        {
                            Some((
                                name,
                                pred.kind().rebind(proj.term.expect_type()),
                                pred.kind().rebind(args.as_slice()),
                            ))
                        } else {
                            None
                        }
                    })
                }
                _ => None,
            })
        else {
            return None;
        };

        let output = self.instantiate_binder_with_fresh_vars(
            DUMMY_SP,
            BoundRegionConversionTime::FnCall,
            output,
        );
        let inputs = inputs
            .skip_binder()
            .iter()
            .map(|ty| {
                self.instantiate_binder_with_fresh_vars(
                    DUMMY_SP,
                    BoundRegionConversionTime::FnCall,
                    inputs.rebind(*ty),
                )
            })
            .collect();

        // We don't want to register any extra obligations, which should be
        // implied by wf, but also because that would possibly result in
        // erroneous errors later on.
        let InferOk { value: output, obligations: _ } =
            self.at(&ObligationCause::dummy(), param_env).normalize(output);

        if output.is_ty_var() { None } else { Some((def_id_or_name, output, inputs)) }
    }

    pub(super) fn suggest_add_reference_to_arg(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
        poly_trait_pred: ty::PolyTraitPredicate<'tcx>,
        has_custom_message: bool,
    ) -> bool {
        let span = obligation.cause.span;

        let code = match obligation.cause.code() {
            ObligationCauseCode::FunctionArg { parent_code, .. } => parent_code,
            // FIXME(compiler-errors): This is kind of a mess, but required for obligations
            // that come from a path expr to affect the *call* expr.
            c @ ObligationCauseCode::WhereClauseInExpr(_, _, hir_id, _)
                if self.tcx.hir_span(*hir_id).lo() == span.lo() =>
            {
                c
            }
            c if matches!(
                span.ctxt().outer_expn_data().kind,
                ExpnKind::Desugaring(DesugaringKind::ForLoop)
            ) =>
            {
                c
            }
            _ => return false,
        };

        // List of traits for which it would be nonsensical to suggest borrowing.
        // For instance, immutable references are always Copy, so suggesting to
        // borrow would always succeed, but it's probably not what the user wanted.
        let mut never_suggest_borrow: Vec<_> =
            [LangItem::Copy, LangItem::Clone, LangItem::Unpin, LangItem::Sized]
                .iter()
                .filter_map(|lang_item| self.tcx.lang_items().get(*lang_item))
                .collect();

        if let Some(def_id) = self.tcx.get_diagnostic_item(sym::Send) {
            never_suggest_borrow.push(def_id);
        }

        let param_env = obligation.param_env;

        // Try to apply the original trait bound by borrowing.
        let mut try_borrowing = |old_pred: ty::PolyTraitPredicate<'tcx>,
                                 blacklist: &[DefId]|
         -> bool {
            if blacklist.contains(&old_pred.def_id()) {
                return false;
            }
            // We map bounds to `&T` and `&mut T`
            let trait_pred_and_imm_ref = old_pred.map_bound(|trait_pred| {
                (
                    trait_pred,
                    Ty::new_imm_ref(self.tcx, self.tcx.lifetimes.re_static, trait_pred.self_ty()),
                )
            });
            let trait_pred_and_mut_ref = old_pred.map_bound(|trait_pred| {
                (
                    trait_pred,
                    Ty::new_mut_ref(self.tcx, self.tcx.lifetimes.re_static, trait_pred.self_ty()),
                )
            });

            let mk_result = |trait_pred_and_new_ty| {
                let obligation =
                    self.mk_trait_obligation_with_new_self_ty(param_env, trait_pred_and_new_ty);
                self.predicate_must_hold_modulo_regions(&obligation)
            };
            let imm_ref_self_ty_satisfies_pred = mk_result(trait_pred_and_imm_ref);
            let mut_ref_self_ty_satisfies_pred = mk_result(trait_pred_and_mut_ref);

            let (ref_inner_ty_satisfies_pred, ref_inner_ty_mut) =
                if let ObligationCauseCode::WhereClauseInExpr(..) = obligation.cause.code()
                    && let ty::Ref(_, ty, mutability) = old_pred.self_ty().skip_binder().kind()
                {
                    (
                        mk_result(old_pred.map_bound(|trait_pred| (trait_pred, *ty))),
                        mutability.is_mut(),
                    )
                } else {
                    (false, false)
                };

            if imm_ref_self_ty_satisfies_pred
                || mut_ref_self_ty_satisfies_pred
                || ref_inner_ty_satisfies_pred
            {
                if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                    // We don't want a borrowing suggestion on the fields in structs,
                    // ```
                    // struct Foo {
                    //  the_foos: Vec<Foo>
                    // }
                    // ```
                    if !matches!(
                        span.ctxt().outer_expn_data().kind,
                        ExpnKind::Root | ExpnKind::Desugaring(DesugaringKind::ForLoop)
                    ) {
                        return false;
                    }
                    if snippet.starts_with('&') {
                        // This is already a literal borrow and the obligation is failing
                        // somewhere else in the obligation chain. Do not suggest non-sense.
                        return false;
                    }
                    // We have a very specific type of error, where just borrowing this argument
                    // might solve the problem. In cases like this, the important part is the
                    // original type obligation, not the last one that failed, which is arbitrary.
                    // Because of this, we modify the error to refer to the original obligation and
                    // return early in the caller.

                    let msg = format!(
                        "the trait bound `{}` is not satisfied",
                        self.tcx.short_string(old_pred, err.long_ty_path()),
                    );
                    let self_ty_str =
                        self.tcx.short_string(old_pred.self_ty().skip_binder(), err.long_ty_path());
                    if has_custom_message {
                        err.note(msg);
                    } else {
                        err.messages = vec![(rustc_errors::DiagMessage::from(msg), Style::NoStyle)];
                    }
                    err.span_label(
                        span,
                        format!(
                            "the trait `{}` is not implemented for `{self_ty_str}`",
                            old_pred.print_modifiers_and_trait_path()
                        ),
                    );

                    if imm_ref_self_ty_satisfies_pred && mut_ref_self_ty_satisfies_pred {
                        err.span_suggestions(
                            span.shrink_to_lo(),
                            "consider borrowing here",
                            ["&".to_string(), "&mut ".to_string()],
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        let is_mut = mut_ref_self_ty_satisfies_pred || ref_inner_ty_mut;
                        let sugg_prefix = format!("&{}", if is_mut { "mut " } else { "" });
                        let sugg_msg = format!(
                            "consider{} borrowing here",
                            if is_mut { " mutably" } else { "" }
                        );

                        // Issue #109436, we need to add parentheses properly for method calls
                        // for example, `foo.into()` should be `(&foo).into()`
                        if let Some(_) =
                            self.tcx.sess.source_map().span_look_ahead(span, ".", Some(50))
                        {
                            err.multipart_suggestion_verbose(
                                sugg_msg,
                                vec![
                                    (span.shrink_to_lo(), format!("({sugg_prefix}")),
                                    (span.shrink_to_hi(), ")".to_string()),
                                ],
                                Applicability::MaybeIncorrect,
                            );
                            return true;
                        }

                        // Issue #104961, we need to add parentheses properly for compound expressions
                        // for example, `x.starts_with("hi".to_string() + "you")`
                        // should be `x.starts_with(&("hi".to_string() + "you"))`
                        let Some(body) = self.tcx.hir_maybe_body_owned_by(obligation.cause.body_id)
                        else {
                            return false;
                        };
                        let mut expr_finder = FindExprBySpan::new(span, self.tcx);
                        expr_finder.visit_expr(body.value);
                        let Some(expr) = expr_finder.result else {
                            return false;
                        };
                        let needs_parens = expr_needs_parens(expr);

                        let span = if needs_parens { span } else { span.shrink_to_lo() };
                        let suggestions = if !needs_parens {
                            vec![(span.shrink_to_lo(), sugg_prefix)]
                        } else {
                            vec![
                                (span.shrink_to_lo(), format!("{sugg_prefix}(")),
                                (span.shrink_to_hi(), ")".to_string()),
                            ]
                        };
                        err.multipart_suggestion_verbose(
                            sugg_msg,
                            suggestions,
                            Applicability::MaybeIncorrect,
                        );
                    }
                    return true;
                }
            }
            return false;
        };

        if let ObligationCauseCode::ImplDerived(cause) = &*code {
            try_borrowing(cause.derived.parent_trait_pred, &[])
        } else if let ObligationCauseCode::WhereClause(..)
        | ObligationCauseCode::WhereClauseInExpr(..) = code
        {
            try_borrowing(poly_trait_pred, &never_suggest_borrow)
        } else {
            false
        }
    }

    // Suggest borrowing the type
    pub(super) fn suggest_borrowing_for_object_cast(
        &self,
        err: &mut Diag<'_>,
        obligation: &PredicateObligation<'tcx>,
        self_ty: Ty<'tcx>,
        target_ty: Ty<'tcx>,
    ) {
        let ty::Ref(_, object_ty, hir::Mutability::Not) = target_ty.kind() else {
            return;
        };
        let ty::Dynamic(predicates, _, ty::Dyn) = object_ty.kind() else {
            return;
        };
        let self_ref_ty = Ty::new_imm_ref(self.tcx, self.tcx.lifetimes.re_erased, self_ty);

        for predicate in predicates.iter() {
            if !self.predicate_must_hold_modulo_regions(
                &obligation.with(self.tcx, predicate.with_self_ty(self.tcx, self_ref_ty)),
            ) {
                return;
            }
        }

        err.span_suggestion_verbose(
            obligation.cause.span.shrink_to_lo(),
            format!(
                "consider borrowing the value, since `&{self_ty}` can be coerced into `{target_ty}`"
            ),
            "&",
            Applicability::MaybeIncorrect,
        );
    }

    /// Whenever references are used by mistake, like `for (i, e) in &vec.iter().enumerate()`,
    /// suggest removing these references until we reach a type that implements the trait.
    pub(super) fn suggest_remove_reference(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        let mut span = obligation.cause.span;
        let mut trait_pred = trait_pred;
        let mut code = obligation.cause.code();
        while let Some((c, Some(parent_trait_pred))) = code.parent_with_predicate() {
            // We want the root obligation, in order to detect properly handle
            // `for _ in &mut &mut vec![] {}`.
            code = c;
            trait_pred = parent_trait_pred;
        }
        while span.desugaring_kind().is_some() {
            // Remove all the hir desugaring contexts while maintaining the macro contexts.
            span.remove_mark();
        }
        let mut expr_finder = super::FindExprBySpan::new(span, self.tcx);
        let Some(body) = self.tcx.hir_maybe_body_owned_by(obligation.cause.body_id) else {
            return false;
        };
        expr_finder.visit_expr(body.value);
        let mut maybe_suggest = |suggested_ty, count, suggestions| {
            // Remapping bound vars here
            let trait_pred_and_suggested_ty =
                trait_pred.map_bound(|trait_pred| (trait_pred, suggested_ty));

            let new_obligation = self.mk_trait_obligation_with_new_self_ty(
                obligation.param_env,
                trait_pred_and_suggested_ty,
            );

            if self.predicate_may_hold(&new_obligation) {
                let msg = if count == 1 {
                    "consider removing the leading `&`-reference".to_string()
                } else {
                    format!("consider removing {count} leading `&`-references")
                };

                err.multipart_suggestion_verbose(
                    msg,
                    suggestions,
                    Applicability::MachineApplicable,
                );
                true
            } else {
                false
            }
        };

        // Maybe suggest removal of borrows from types in type parameters, like in
        // `src/test/ui/not-panic/not-panic-safe.rs`.
        let mut count = 0;
        let mut suggestions = vec![];
        // Skipping binder here, remapping below
        let mut suggested_ty = trait_pred.self_ty().skip_binder();
        if let Some(mut hir_ty) = expr_finder.ty_result {
            while let hir::TyKind::Ref(_, mut_ty) = &hir_ty.kind {
                count += 1;
                let span = hir_ty.span.until(mut_ty.ty.span);
                suggestions.push((span, String::new()));

                let ty::Ref(_, inner_ty, _) = suggested_ty.kind() else {
                    break;
                };
                suggested_ty = *inner_ty;

                hir_ty = mut_ty.ty;

                if maybe_suggest(suggested_ty, count, suggestions.clone()) {
                    return true;
                }
            }
        }

        // Maybe suggest removal of borrows from expressions, like in `for i in &&&foo {}`.
        let Some(mut expr) = expr_finder.result else {
            return false;
        };
        let mut count = 0;
        let mut suggestions = vec![];
        // Skipping binder here, remapping below
        let mut suggested_ty = trait_pred.self_ty().skip_binder();
        'outer: loop {
            while let hir::ExprKind::AddrOf(_, _, borrowed) = expr.kind {
                count += 1;
                let span = if expr.span.eq_ctxt(borrowed.span) {
                    expr.span.until(borrowed.span)
                } else {
                    expr.span.with_hi(expr.span.lo() + BytePos(1))
                };

                match self.tcx.sess.source_map().span_to_snippet(span) {
                    Ok(snippet) if snippet.starts_with("&") => {}
                    _ => break 'outer,
                }

                suggestions.push((span, String::new()));

                let ty::Ref(_, inner_ty, _) = suggested_ty.kind() else {
                    break 'outer;
                };
                suggested_ty = *inner_ty;

                expr = borrowed;

                if maybe_suggest(suggested_ty, count, suggestions.clone()) {
                    return true;
                }
            }
            if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = expr.kind
                && let Res::Local(hir_id) = path.res
                && let hir::Node::Pat(binding) = self.tcx.hir_node(hir_id)
                && let hir::Node::LetStmt(local) = self.tcx.parent_hir_node(binding.hir_id)
                && let None = local.ty
                && let Some(binding_expr) = local.init
            {
                expr = binding_expr;
            } else {
                break 'outer;
            }
        }
        false
    }

    pub(super) fn suggest_remove_await(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
    ) {
        if let ObligationCauseCode::AwaitableExpr(hir_id) = obligation.cause.code().peel_derives()
            && let hir::Node::Expr(expr) = self.tcx.hir_node(*hir_id)
        {
            // FIXME: use `obligation.predicate.kind()...trait_ref.self_ty()` to see if we have `()`
            // and if not maybe suggest doing something else? If we kept the expression around we
            // could also check if it is an fn call (very likely) and suggest changing *that*, if
            // it is from the local crate.

            // use nth(1) to skip one layer of desugaring from `IntoIter::into_iter`
            if let Some((_, hir::Node::Expr(await_expr))) = self.tcx.hir_parent_iter(*hir_id).nth(1)
                && let Some(expr_span) = expr.span.find_ancestor_inside_same_ctxt(await_expr.span)
            {
                let removal_span = self
                    .tcx
                    .sess
                    .source_map()
                    .span_extend_while_whitespace(expr_span)
                    .shrink_to_hi()
                    .to(await_expr.span.shrink_to_hi());
                err.span_suggestion_verbose(
                    removal_span,
                    "remove the `.await`",
                    "",
                    Applicability::MachineApplicable,
                );
            } else {
                err.span_label(obligation.cause.span, "remove the `.await`");
            }
            // FIXME: account for associated `async fn`s.
            if let hir::Expr { span, kind: hir::ExprKind::Call(base, _), .. } = expr {
                if let ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) =
                    obligation.predicate.kind().skip_binder()
                {
                    err.span_label(*span, format!("this call returns `{}`", pred.self_ty()));
                }
                if let Some(typeck_results) = &self.typeck_results
                    && let ty = typeck_results.expr_ty_adjusted(base)
                    && let ty::FnDef(def_id, _args) = ty.kind()
                    && let Some(hir::Node::Item(item)) = self.tcx.hir_get_if_local(*def_id)
                {
                    let (ident, _, _, _) = item.expect_fn();
                    let msg = format!("alternatively, consider making `fn {ident}` asynchronous");
                    if item.vis_span.is_empty() {
                        err.span_suggestion_verbose(
                            item.span.shrink_to_lo(),
                            msg,
                            "async ",
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        err.span_suggestion_verbose(
                            item.vis_span.shrink_to_hi(),
                            msg,
                            " async",
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }
        }
    }

    /// Check if the trait bound is implemented for a different mutability and note it in the
    /// final error.
    pub(super) fn suggest_change_mut(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) {
        let points_at_arg =
            matches!(obligation.cause.code(), ObligationCauseCode::FunctionArg { .. },);

        let span = obligation.cause.span;
        if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
            let refs_number =
                snippet.chars().filter(|c| !c.is_whitespace()).take_while(|c| *c == '&').count();
            if let Some('\'') = snippet.chars().filter(|c| !c.is_whitespace()).nth(refs_number) {
                // Do not suggest removal of borrow from type arguments.
                return;
            }
            let trait_pred = self.resolve_vars_if_possible(trait_pred);
            if trait_pred.has_non_region_infer() {
                // Do not ICE while trying to find if a reborrow would succeed on a trait with
                // unresolved bindings.
                return;
            }

            // Skipping binder here, remapping below
            if let ty::Ref(region, t_type, mutability) = *trait_pred.skip_binder().self_ty().kind()
            {
                let suggested_ty = match mutability {
                    hir::Mutability::Mut => Ty::new_imm_ref(self.tcx, region, t_type),
                    hir::Mutability::Not => Ty::new_mut_ref(self.tcx, region, t_type),
                };

                // Remapping bound vars here
                let trait_pred_and_suggested_ty =
                    trait_pred.map_bound(|trait_pred| (trait_pred, suggested_ty));

                let new_obligation = self.mk_trait_obligation_with_new_self_ty(
                    obligation.param_env,
                    trait_pred_and_suggested_ty,
                );
                let suggested_ty_would_satisfy_obligation = self
                    .evaluate_obligation_no_overflow(&new_obligation)
                    .must_apply_modulo_regions();
                if suggested_ty_would_satisfy_obligation {
                    let sp = self
                        .tcx
                        .sess
                        .source_map()
                        .span_take_while(span, |c| c.is_whitespace() || *c == '&');
                    if points_at_arg && mutability.is_not() && refs_number > 0 {
                        // If we have a call like foo(&mut buf), then don't suggest foo(&mut mut buf)
                        if snippet
                            .trim_start_matches(|c: char| c.is_whitespace() || c == '&')
                            .starts_with("mut")
                        {
                            return;
                        }
                        err.span_suggestion_verbose(
                            sp,
                            "consider changing this borrow's mutability",
                            "&mut ",
                            Applicability::MachineApplicable,
                        );
                    } else {
                        err.note(format!(
                            "`{}` is implemented for `{}`, but not for `{}`",
                            trait_pred.print_modifiers_and_trait_path(),
                            suggested_ty,
                            trait_pred.skip_binder().self_ty(),
                        ));
                    }
                }
            }
        }
    }

    pub(super) fn suggest_semicolon_removal(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
        span: Span,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        let node = self.tcx.hir_node_by_def_id(obligation.cause.body_id);
        if let hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn {sig, body: body_id, .. }, .. }) = node
            && let hir::ExprKind::Block(blk, _) = &self.tcx.hir_body(*body_id).value.kind
            && sig.decl.output.span().overlaps(span)
            && blk.expr.is_none()
            && trait_pred.self_ty().skip_binder().is_unit()
            && let Some(stmt) = blk.stmts.last()
            && let hir::StmtKind::Semi(expr) = stmt.kind
            // Only suggest this if the expression behind the semicolon implements the predicate
            && let Some(typeck_results) = &self.typeck_results
            && let Some(ty) = typeck_results.expr_ty_opt(expr)
            && self.predicate_may_hold(&self.mk_trait_obligation_with_new_self_ty(
                obligation.param_env, trait_pred.map_bound(|trait_pred| (trait_pred, ty))
            ))
        {
            err.span_label(
                expr.span,
                format!(
                    "this expression has type `{}`, which implements `{}`",
                    ty,
                    trait_pred.print_modifiers_and_trait_path()
                ),
            );
            err.span_suggestion(
                self.tcx.sess.source_map().end_point(stmt.span),
                "remove this semicolon",
                "",
                Applicability::MachineApplicable,
            );
            return true;
        }
        false
    }

    pub(super) fn return_type_span(&self, obligation: &PredicateObligation<'tcx>) -> Option<Span> {
        let hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn { sig, .. }, .. }) =
            self.tcx.hir_node_by_def_id(obligation.cause.body_id)
        else {
            return None;
        };

        if let hir::FnRetTy::Return(ret_ty) = sig.decl.output { Some(ret_ty.span) } else { None }
    }

    /// If all conditions are met to identify a returned `dyn Trait`, suggest using `impl Trait` if
    /// applicable and signal that the error has been expanded appropriately and needs to be
    /// emitted.
    pub(super) fn suggest_impl_trait(
        &self,
        err: &mut Diag<'_>,
        obligation: &PredicateObligation<'tcx>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        let ObligationCauseCode::SizedReturnType = obligation.cause.code() else {
            return false;
        };
        let ty::Dynamic(_, _, ty::Dyn) = trait_pred.self_ty().skip_binder().kind() else {
            return false;
        };

        err.code(E0746);
        err.primary_message("return type cannot be a trait object without pointer indirection");
        err.children.clear();

        let span = obligation.cause.span;
        let body = self.tcx.hir_body_owned_by(obligation.cause.body_id);

        let mut visitor = ReturnsVisitor::default();
        visitor.visit_body(&body);

        let (pre, impl_span) = if let Ok(snip) = self.tcx.sess.source_map().span_to_snippet(span)
            && snip.starts_with("dyn ")
        {
            ("", span.with_hi(span.lo() + BytePos(4)))
        } else {
            ("dyn ", span.shrink_to_lo())
        };

        err.span_suggestion_verbose(
            impl_span,
            "consider returning an `impl Trait` instead of a `dyn Trait`",
            "impl ",
            Applicability::MaybeIncorrect,
        );

        let mut sugg = vec![
            (span.shrink_to_lo(), format!("Box<{pre}")),
            (span.shrink_to_hi(), ">".to_string()),
        ];
        sugg.extend(visitor.returns.into_iter().flat_map(|expr| {
            let span =
                expr.span.find_ancestor_in_same_ctxt(obligation.cause.span).unwrap_or(expr.span);
            if !span.can_be_used_for_suggestions() {
                vec![]
            } else if let hir::ExprKind::Call(path, ..) = expr.kind
                && let hir::ExprKind::Path(hir::QPath::TypeRelative(ty, method)) = path.kind
                && method.ident.name == sym::new
                && let hir::TyKind::Path(hir::QPath::Resolved(.., box_path)) = ty.kind
                && box_path
                    .res
                    .opt_def_id()
                    .is_some_and(|def_id| self.tcx.is_lang_item(def_id, LangItem::OwnedBox))
            {
                // Don't box `Box::new`
                vec![]
            } else {
                vec![
                    (span.shrink_to_lo(), "Box::new(".to_string()),
                    (span.shrink_to_hi(), ")".to_string()),
                ]
            }
        }));

        err.multipart_suggestion(
            format!(
                "alternatively, box the return type, and wrap all of the returned values in \
                 `Box::new`",
            ),
            sugg,
            Applicability::MaybeIncorrect,
        );

        true
    }

    pub(super) fn report_closure_arg_mismatch(
        &self,
        span: Span,
        found_span: Option<Span>,
        found: ty::TraitRef<'tcx>,
        expected: ty::TraitRef<'tcx>,
        cause: &ObligationCauseCode<'tcx>,
        found_node: Option<Node<'_>>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Diag<'a> {
        pub(crate) fn build_fn_sig_ty<'tcx>(
            infcx: &InferCtxt<'tcx>,
            trait_ref: ty::TraitRef<'tcx>,
        ) -> Ty<'tcx> {
            let inputs = trait_ref.args.type_at(1);
            let sig = match inputs.kind() {
                ty::Tuple(inputs) if infcx.tcx.is_fn_trait(trait_ref.def_id) => {
                    infcx.tcx.mk_fn_sig(
                        *inputs,
                        infcx.next_ty_var(DUMMY_SP),
                        false,
                        hir::Safety::Safe,
                        ExternAbi::Rust,
                    )
                }
                _ => infcx.tcx.mk_fn_sig(
                    [inputs],
                    infcx.next_ty_var(DUMMY_SP),
                    false,
                    hir::Safety::Safe,
                    ExternAbi::Rust,
                ),
            };

            Ty::new_fn_ptr(infcx.tcx, ty::Binder::dummy(sig))
        }

        let argument_kind = match expected.self_ty().kind() {
            ty::Closure(..) => "closure",
            ty::Coroutine(..) => "coroutine",
            _ => "function",
        };
        let mut err = struct_span_code_err!(
            self.dcx(),
            span,
            E0631,
            "type mismatch in {argument_kind} arguments",
        );

        err.span_label(span, "expected due to this");

        let found_span = found_span.unwrap_or(span);
        err.span_label(found_span, "found signature defined here");

        let expected = build_fn_sig_ty(self, expected);
        let found = build_fn_sig_ty(self, found);

        let (expected_str, found_str) = self.cmp(expected, found);

        let signature_kind = format!("{argument_kind} signature");
        err.note_expected_found(&signature_kind, expected_str, &signature_kind, found_str);

        self.note_conflicting_fn_args(&mut err, cause, expected, found, param_env);
        self.note_conflicting_closure_bounds(cause, &mut err);

        if let Some(found_node) = found_node {
            hint_missing_borrow(self, param_env, span, found, expected, found_node, &mut err);
        }

        err
    }

    fn note_conflicting_fn_args(
        &self,
        err: &mut Diag<'_>,
        cause: &ObligationCauseCode<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) {
        let ObligationCauseCode::FunctionArg { arg_hir_id, .. } = cause else {
            return;
        };
        let ty::FnPtr(sig_tys, hdr) = expected.kind() else {
            return;
        };
        let expected = sig_tys.with(*hdr);
        let ty::FnPtr(sig_tys, hdr) = found.kind() else {
            return;
        };
        let found = sig_tys.with(*hdr);
        let Node::Expr(arg) = self.tcx.hir_node(*arg_hir_id) else {
            return;
        };
        let hir::ExprKind::Path(path) = arg.kind else {
            return;
        };
        let expected_inputs = self.tcx.instantiate_bound_regions_with_erased(expected).inputs();
        let found_inputs = self.tcx.instantiate_bound_regions_with_erased(found).inputs();
        let both_tys = expected_inputs.iter().copied().zip(found_inputs.iter().copied());

        let arg_expr = |infcx: &InferCtxt<'tcx>, name, expected: Ty<'tcx>, found: Ty<'tcx>| {
            let (expected_ty, expected_refs) = get_deref_type_and_refs(expected);
            let (found_ty, found_refs) = get_deref_type_and_refs(found);

            if infcx.can_eq(param_env, found_ty, expected_ty) {
                if found_refs.len() == expected_refs.len()
                    && found_refs.iter().eq(expected_refs.iter())
                {
                    name
                } else if found_refs.len() > expected_refs.len() {
                    let refs = &found_refs[..found_refs.len() - expected_refs.len()];
                    if found_refs[..expected_refs.len()].iter().eq(expected_refs.iter()) {
                        format!(
                            "{}{name}",
                            refs.iter()
                                .map(|mutbl| format!("&{}", mutbl.prefix_str()))
                                .collect::<Vec<_>>()
                                .join(""),
                        )
                    } else {
                        // The refs have different mutability.
                        format!(
                            "{}*{name}",
                            refs.iter()
                                .map(|mutbl| format!("&{}", mutbl.prefix_str()))
                                .collect::<Vec<_>>()
                                .join(""),
                        )
                    }
                } else if expected_refs.len() > found_refs.len() {
                    format!(
                        "{}{name}",
                        (0..(expected_refs.len() - found_refs.len()))
                            .map(|_| "*")
                            .collect::<Vec<_>>()
                            .join(""),
                    )
                } else {
                    format!(
                        "{}{name}",
                        found_refs
                            .iter()
                            .map(|mutbl| format!("&{}", mutbl.prefix_str()))
                            .chain(found_refs.iter().map(|_| "*".to_string()))
                            .collect::<Vec<_>>()
                            .join(""),
                    )
                }
            } else {
                format!("/* {found} */")
            }
        };
        let args_have_same_underlying_type = both_tys.clone().all(|(expected, found)| {
            let (expected_ty, _) = get_deref_type_and_refs(expected);
            let (found_ty, _) = get_deref_type_and_refs(found);
            self.can_eq(param_env, found_ty, expected_ty)
        });
        let (closure_names, call_names): (Vec<_>, Vec<_>) = if args_have_same_underlying_type
            && !expected_inputs.is_empty()
            && expected_inputs.len() == found_inputs.len()
            && let Some(typeck) = &self.typeck_results
            && let Res::Def(res_kind, fn_def_id) = typeck.qpath_res(&path, *arg_hir_id)
            && res_kind.is_fn_like()
        {
            let closure: Vec<_> = self
                .tcx
                .fn_arg_idents(fn_def_id)
                .iter()
                .enumerate()
                .map(|(i, ident)| {
                    if let Some(ident) = ident
                        && !matches!(ident, Ident { name: kw::Underscore | kw::SelfLower, .. })
                    {
                        format!("{ident}")
                    } else {
                        format!("arg{i}")
                    }
                })
                .collect();
            let args = closure
                .iter()
                .zip(both_tys)
                .map(|(name, (expected, found))| {
                    arg_expr(self.infcx, name.to_owned(), expected, found)
                })
                .collect();
            (closure, args)
        } else {
            let closure_args = expected_inputs
                .iter()
                .enumerate()
                .map(|(i, _)| format!("arg{i}"))
                .collect::<Vec<_>>();
            let call_args = both_tys
                .enumerate()
                .map(|(i, (expected, found))| {
                    arg_expr(self.infcx, format!("arg{i}"), expected, found)
                })
                .collect::<Vec<_>>();
            (closure_args, call_args)
        };
        let closure_names: Vec<_> = closure_names
            .into_iter()
            .zip(expected_inputs.iter())
            .map(|(name, ty)| {
                format!(
                    "{name}{}",
                    if ty.has_infer_types() {
                        String::new()
                    } else if ty.references_error() {
                        ": /* type */".to_string()
                    } else {
                        format!(": {ty}")
                    }
                )
            })
            .collect();
        err.multipart_suggestion(
            "consider wrapping the function in a closure",
            vec![
                (arg.span.shrink_to_lo(), format!("|{}| ", closure_names.join(", "))),
                (arg.span.shrink_to_hi(), format!("({})", call_names.join(", "))),
            ],
            Applicability::MaybeIncorrect,
        );
    }

    // Add a note if there are two `Fn`-family bounds that have conflicting argument
    // requirements, which will always cause a closure to have a type error.
    fn note_conflicting_closure_bounds(
        &self,
        cause: &ObligationCauseCode<'tcx>,
        err: &mut Diag<'_>,
    ) {
        // First, look for an `WhereClauseInExpr`, which means we can get
        // the uninstantiated predicate list of the called function. And check
        // that the predicate that we failed to satisfy is a `Fn`-like trait.
        if let ObligationCauseCode::WhereClauseInExpr(def_id, _, _, idx) = cause
            && let predicates = self.tcx.predicates_of(def_id).instantiate_identity(self.tcx)
            && let Some(pred) = predicates.predicates.get(*idx)
            && let ty::ClauseKind::Trait(trait_pred) = pred.kind().skip_binder()
            && self.tcx.is_fn_trait(trait_pred.def_id())
        {
            let expected_self =
                self.tcx.anonymize_bound_vars(pred.kind().rebind(trait_pred.self_ty()));
            let expected_args =
                self.tcx.anonymize_bound_vars(pred.kind().rebind(trait_pred.trait_ref.args));

            // Find another predicate whose self-type is equal to the expected self type,
            // but whose args don't match.
            let other_pred = predicates.into_iter().enumerate().find(|(other_idx, (pred, _))| {
                match pred.kind().skip_binder() {
                    ty::ClauseKind::Trait(trait_pred)
                        if self.tcx.is_fn_trait(trait_pred.def_id())
                            && other_idx != idx
                            // Make sure that the self type matches
                            // (i.e. constraining this closure)
                            && expected_self
                                == self.tcx.anonymize_bound_vars(
                                    pred.kind().rebind(trait_pred.self_ty()),
                                )
                            // But the args don't match (i.e. incompatible args)
                            && expected_args
                                != self.tcx.anonymize_bound_vars(
                                    pred.kind().rebind(trait_pred.trait_ref.args),
                                ) =>
                    {
                        true
                    }
                    _ => false,
                }
            });
            // If we found one, then it's very likely the cause of the error.
            if let Some((_, (_, other_pred_span))) = other_pred {
                err.span_note(
                    other_pred_span,
                    "closure inferred to have a different signature due to this bound",
                );
            }
        }
    }

    pub(super) fn suggest_fully_qualified_path(
        &self,
        err: &mut Diag<'_>,
        item_def_id: DefId,
        span: Span,
        trait_ref: DefId,
    ) {
        if let Some(assoc_item) = self.tcx.opt_associated_item(item_def_id) {
            if let ty::AssocKind::Const { .. } | ty::AssocKind::Type { .. } = assoc_item.kind {
                err.note(format!(
                    "{}s cannot be accessed directly on a `trait`, they can only be \
                        accessed through a specific `impl`",
                    self.tcx.def_kind_descr(assoc_item.as_def_kind(), item_def_id)
                ));

                if !assoc_item.is_impl_trait_in_trait() {
                    err.span_suggestion_verbose(
                        span,
                        "use the fully qualified path to an implementation",
                        format!(
                            "<Type as {}>::{}",
                            self.tcx.def_path_str(trait_ref),
                            assoc_item.name()
                        ),
                        Applicability::HasPlaceholders,
                    );
                }
            }
        }
    }

    /// Adds an async-await specific note to the diagnostic when the future does not implement
    /// an auto trait because of a captured type.
    ///
    /// ```text
    /// note: future does not implement `Qux` as this value is used across an await
    ///   --> $DIR/issue-64130-3-other.rs:17:5
    ///    |
    /// LL |     let x = Foo;
    ///    |         - has type `Foo`
    /// LL |     baz().await;
    ///    |     ^^^^^^^^^^^ await occurs here, with `x` maybe used later
    /// LL | }
    ///    | - `x` is later dropped here
    /// ```
    ///
    /// When the diagnostic does not implement `Send` or `Sync` specifically, then the diagnostic
    /// is "replaced" with a different message and a more specific error.
    ///
    /// ```text
    /// error: future cannot be sent between threads safely
    ///   --> $DIR/issue-64130-2-send.rs:21:5
    ///    |
    /// LL | fn is_send<T: Send>(t: T) { }
    ///    |               ---- required by this bound in `is_send`
    /// ...
    /// LL |     is_send(bar());
    ///    |     ^^^^^^^ future returned by `bar` is not send
    ///    |
    ///    = help: within `impl std::future::Future`, the trait `std::marker::Send` is not
    ///            implemented for `Foo`
    /// note: future is not send as this value is used across an await
    ///   --> $DIR/issue-64130-2-send.rs:15:5
    ///    |
    /// LL |     let x = Foo;
    ///    |         - has type `Foo`
    /// LL |     baz().await;
    ///    |     ^^^^^^^^^^^ await occurs here, with `x` maybe used later
    /// LL | }
    ///    | - `x` is later dropped here
    /// ```
    ///
    /// Returns `true` if an async-await specific note was added to the diagnostic.
    #[instrument(level = "debug", skip_all, fields(?obligation.predicate, ?obligation.cause.span))]
    pub fn maybe_note_obligation_cause_for_async_await<G: EmissionGuarantee>(
        &self,
        err: &mut Diag<'_, G>,
        obligation: &PredicateObligation<'tcx>,
    ) -> bool {
        // Attempt to detect an async-await error by looking at the obligation causes, looking
        // for a coroutine to be present.
        //
        // When a future does not implement a trait because of a captured type in one of the
        // coroutines somewhere in the call stack, then the result is a chain of obligations.
        //
        // Given an `async fn` A that calls an `async fn` B which captures a non-send type and that
        // future is passed as an argument to a function C which requires a `Send` type, then the
        // chain looks something like this:
        //
        // - `BuiltinDerivedObligation` with a coroutine witness (B)
        // - `BuiltinDerivedObligation` with a coroutine (B)
        // - `BuiltinDerivedObligation` with `impl std::future::Future` (B)
        // - `BuiltinDerivedObligation` with a coroutine witness (A)
        // - `BuiltinDerivedObligation` with a coroutine (A)
        // - `BuiltinDerivedObligation` with `impl std::future::Future` (A)
        // - `BindingObligation` with `impl_send` (Send requirement)
        //
        // The first obligation in the chain is the most useful and has the coroutine that captured
        // the type. The last coroutine (`outer_coroutine` below) has information about where the
        // bound was introduced. At least one coroutine should be present for this diagnostic to be
        // modified.
        let (mut trait_ref, mut target_ty) = match obligation.predicate.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(p)) => (Some(p), Some(p.self_ty())),
            _ => (None, None),
        };
        let mut coroutine = None;
        let mut outer_coroutine = None;
        let mut next_code = Some(obligation.cause.code());

        let mut seen_upvar_tys_infer_tuple = false;

        while let Some(code) = next_code {
            debug!(?code);
            match code {
                ObligationCauseCode::FunctionArg { parent_code, .. } => {
                    next_code = Some(parent_code);
                }
                ObligationCauseCode::ImplDerived(cause) => {
                    let ty = cause.derived.parent_trait_pred.skip_binder().self_ty();
                    debug!(
                        parent_trait_ref = ?cause.derived.parent_trait_pred,
                        self_ty.kind = ?ty.kind(),
                        "ImplDerived",
                    );

                    match *ty.kind() {
                        ty::Coroutine(did, ..) | ty::CoroutineWitness(did, _) => {
                            coroutine = coroutine.or(Some(did));
                            outer_coroutine = Some(did);
                        }
                        ty::Tuple(_) if !seen_upvar_tys_infer_tuple => {
                            // By introducing a tuple of upvar types into the chain of obligations
                            // of a coroutine, the first non-coroutine item is now the tuple itself,
                            // we shall ignore this.

                            seen_upvar_tys_infer_tuple = true;
                        }
                        _ if coroutine.is_none() => {
                            trait_ref = Some(cause.derived.parent_trait_pred.skip_binder());
                            target_ty = Some(ty);
                        }
                        _ => {}
                    }

                    next_code = Some(&cause.derived.parent_code);
                }
                ObligationCauseCode::WellFormedDerived(derived_obligation)
                | ObligationCauseCode::BuiltinDerived(derived_obligation) => {
                    let ty = derived_obligation.parent_trait_pred.skip_binder().self_ty();
                    debug!(
                        parent_trait_ref = ?derived_obligation.parent_trait_pred,
                        self_ty.kind = ?ty.kind(),
                    );

                    match *ty.kind() {
                        ty::Coroutine(did, ..) | ty::CoroutineWitness(did, ..) => {
                            coroutine = coroutine.or(Some(did));
                            outer_coroutine = Some(did);
                        }
                        ty::Tuple(_) if !seen_upvar_tys_infer_tuple => {
                            // By introducing a tuple of upvar types into the chain of obligations
                            // of a coroutine, the first non-coroutine item is now the tuple itself,
                            // we shall ignore this.

                            seen_upvar_tys_infer_tuple = true;
                        }
                        _ if coroutine.is_none() => {
                            trait_ref = Some(derived_obligation.parent_trait_pred.skip_binder());
                            target_ty = Some(ty);
                        }
                        _ => {}
                    }

                    next_code = Some(&derived_obligation.parent_code);
                }
                _ => break,
            }
        }

        // Only continue if a coroutine was found.
        debug!(?coroutine, ?trait_ref, ?target_ty);
        let (Some(coroutine_did), Some(trait_ref), Some(target_ty)) =
            (coroutine, trait_ref, target_ty)
        else {
            return false;
        };

        let span = self.tcx.def_span(coroutine_did);

        let coroutine_did_root = self.tcx.typeck_root_def_id(coroutine_did);
        debug!(
            ?coroutine_did,
            ?coroutine_did_root,
            typeck_results.hir_owner = ?self.typeck_results.as_ref().map(|t| t.hir_owner),
            ?span,
        );

        let coroutine_body =
            coroutine_did.as_local().and_then(|def_id| self.tcx.hir_maybe_body_owned_by(def_id));
        let mut visitor = AwaitsVisitor::default();
        if let Some(body) = coroutine_body {
            visitor.visit_body(&body);
        }
        debug!(awaits = ?visitor.awaits);

        // Look for a type inside the coroutine interior that matches the target type to get
        // a span.
        let target_ty_erased = self.tcx.erase_regions(target_ty);
        let ty_matches = |ty| -> bool {
            // Careful: the regions for types that appear in the
            // coroutine interior are not generally known, so we
            // want to erase them when comparing (and anyway,
            // `Send` and other bounds are generally unaffected by
            // the choice of region). When erasing regions, we
            // also have to erase late-bound regions. This is
            // because the types that appear in the coroutine
            // interior generally contain "bound regions" to
            // represent regions that are part of the suspended
            // coroutine frame. Bound regions are preserved by
            // `erase_regions` and so we must also call
            // `instantiate_bound_regions_with_erased`.
            let ty_erased = self.tcx.instantiate_bound_regions_with_erased(ty);
            let ty_erased = self.tcx.erase_regions(ty_erased);
            let eq = ty_erased == target_ty_erased;
            debug!(?ty_erased, ?target_ty_erased, ?eq);
            eq
        };

        // Get the typeck results from the infcx if the coroutine is the function we are currently
        // type-checking; otherwise, get them by performing a query. This is needed to avoid
        // cycles. If we can't use resolved types because the coroutine comes from another crate,
        // we still provide a targeted error but without all the relevant spans.
        let coroutine_data = match &self.typeck_results {
            Some(t) if t.hir_owner.to_def_id() == coroutine_did_root => CoroutineData(t),
            _ if coroutine_did.is_local() => {
                CoroutineData(self.tcx.typeck(coroutine_did.expect_local()))
            }
            _ => return false,
        };

        let coroutine_within_in_progress_typeck = match &self.typeck_results {
            Some(t) => t.hir_owner.to_def_id() == coroutine_did_root,
            _ => false,
        };

        let mut interior_or_upvar_span = None;

        let from_awaited_ty = coroutine_data.get_from_await_ty(visitor, self.tcx, ty_matches);
        debug!(?from_awaited_ty);

        // Avoid disclosing internal information to downstream crates.
        if coroutine_did.is_local()
            // Try to avoid cycles.
            && !coroutine_within_in_progress_typeck
            && let Some(coroutine_info) = self.tcx.mir_coroutine_witnesses(coroutine_did)
        {
            debug!(?coroutine_info);
            'find_source: for (variant, source_info) in
                coroutine_info.variant_fields.iter().zip(&coroutine_info.variant_source_info)
            {
                debug!(?variant);
                for &local in variant {
                    let decl = &coroutine_info.field_tys[local];
                    debug!(?decl);
                    if ty_matches(ty::Binder::dummy(decl.ty)) && !decl.ignore_for_traits {
                        interior_or_upvar_span = Some(CoroutineInteriorOrUpvar::Interior(
                            decl.source_info.span,
                            Some((source_info.span, from_awaited_ty)),
                        ));
                        break 'find_source;
                    }
                }
            }
        }

        if interior_or_upvar_span.is_none() {
            interior_or_upvar_span =
                coroutine_data.try_get_upvar_span(self, coroutine_did, ty_matches);
        }

        if interior_or_upvar_span.is_none() && !coroutine_did.is_local() {
            interior_or_upvar_span = Some(CoroutineInteriorOrUpvar::Interior(span, None));
        }

        debug!(?interior_or_upvar_span);
        if let Some(interior_or_upvar_span) = interior_or_upvar_span {
            let is_async = self.tcx.coroutine_is_async(coroutine_did);
            self.note_obligation_cause_for_async_await(
                err,
                interior_or_upvar_span,
                is_async,
                outer_coroutine,
                trait_ref,
                target_ty,
                obligation,
                next_code,
            );
            true
        } else {
            false
        }
    }

    /// Unconditionally adds the diagnostic note described in
    /// `maybe_note_obligation_cause_for_async_await`'s documentation comment.
    #[instrument(level = "debug", skip_all)]
    fn note_obligation_cause_for_async_await<G: EmissionGuarantee>(
        &self,
        err: &mut Diag<'_, G>,
        interior_or_upvar_span: CoroutineInteriorOrUpvar,
        is_async: bool,
        outer_coroutine: Option<DefId>,
        trait_pred: ty::TraitPredicate<'tcx>,
        target_ty: Ty<'tcx>,
        obligation: &PredicateObligation<'tcx>,
        next_code: Option<&ObligationCauseCode<'tcx>>,
    ) {
        let source_map = self.tcx.sess.source_map();

        let (await_or_yield, an_await_or_yield) =
            if is_async { ("await", "an await") } else { ("yield", "a yield") };
        let future_or_coroutine = if is_async { "future" } else { "coroutine" };

        // Special case the primary error message when send or sync is the trait that was
        // not implemented.
        let trait_explanation = if let Some(name @ (sym::Send | sym::Sync)) =
            self.tcx.get_diagnostic_name(trait_pred.def_id())
        {
            let (trait_name, trait_verb) =
                if name == sym::Send { ("`Send`", "sent") } else { ("`Sync`", "shared") };

            err.code = None;
            err.primary_message(format!(
                "{future_or_coroutine} cannot be {trait_verb} between threads safely"
            ));

            let original_span = err.span.primary_span().unwrap();
            let mut span = MultiSpan::from_span(original_span);

            let message = outer_coroutine
                .and_then(|coroutine_did| {
                    Some(match self.tcx.coroutine_kind(coroutine_did).unwrap() {
                        CoroutineKind::Coroutine(_) => format!("coroutine is not {trait_name}"),
                        CoroutineKind::Desugared(
                            CoroutineDesugaring::Async,
                            CoroutineSource::Fn,
                        ) => self
                            .tcx
                            .parent(coroutine_did)
                            .as_local()
                            .map(|parent_did| self.tcx.local_def_id_to_hir_id(parent_did))
                            .and_then(|parent_hir_id| self.tcx.hir_opt_name(parent_hir_id))
                            .map(|name| {
                                format!("future returned by `{name}` is not {trait_name}")
                            })?,
                        CoroutineKind::Desugared(
                            CoroutineDesugaring::Async,
                            CoroutineSource::Block,
                        ) => {
                            format!("future created by async block is not {trait_name}")
                        }
                        CoroutineKind::Desugared(
                            CoroutineDesugaring::Async,
                            CoroutineSource::Closure,
                        ) => {
                            format!("future created by async closure is not {trait_name}")
                        }
                        CoroutineKind::Desugared(
                            CoroutineDesugaring::AsyncGen,
                            CoroutineSource::Fn,
                        ) => self
                            .tcx
                            .parent(coroutine_did)
                            .as_local()
                            .map(|parent_did| self.tcx.local_def_id_to_hir_id(parent_did))
                            .and_then(|parent_hir_id| self.tcx.hir_opt_name(parent_hir_id))
                            .map(|name| {
                                format!("async iterator returned by `{name}` is not {trait_name}")
                            })?,
                        CoroutineKind::Desugared(
                            CoroutineDesugaring::AsyncGen,
                            CoroutineSource::Block,
                        ) => {
                            format!("async iterator created by async gen block is not {trait_name}")
                        }
                        CoroutineKind::Desugared(
                            CoroutineDesugaring::AsyncGen,
                            CoroutineSource::Closure,
                        ) => {
                            format!(
                                "async iterator created by async gen closure is not {trait_name}"
                            )
                        }
                        CoroutineKind::Desugared(CoroutineDesugaring::Gen, CoroutineSource::Fn) => {
                            self.tcx
                                .parent(coroutine_did)
                                .as_local()
                                .map(|parent_did| self.tcx.local_def_id_to_hir_id(parent_did))
                                .and_then(|parent_hir_id| self.tcx.hir_opt_name(parent_hir_id))
                                .map(|name| {
                                    format!("iterator returned by `{name}` is not {trait_name}")
                                })?
                        }
                        CoroutineKind::Desugared(
                            CoroutineDesugaring::Gen,
                            CoroutineSource::Block,
                        ) => {
                            format!("iterator created by gen block is not {trait_name}")
                        }
                        CoroutineKind::Desugared(
                            CoroutineDesugaring::Gen,
                            CoroutineSource::Closure,
                        ) => {
                            format!("iterator created by gen closure is not {trait_name}")
                        }
                    })
                })
                .unwrap_or_else(|| format!("{future_or_coroutine} is not {trait_name}"));

            span.push_span_label(original_span, message);
            err.span(span);

            format!("is not {trait_name}")
        } else {
            format!("does not implement `{}`", trait_pred.print_modifiers_and_trait_path())
        };

        let mut explain_yield = |interior_span: Span, yield_span: Span| {
            let mut span = MultiSpan::from_span(yield_span);
            let snippet = match source_map.span_to_snippet(interior_span) {
                // #70935: If snippet contains newlines, display "the value" instead
                // so that we do not emit complex diagnostics.
                Ok(snippet) if !snippet.contains('\n') => format!("`{snippet}`"),
                _ => "the value".to_string(),
            };
            // note: future is not `Send` as this value is used across an await
            //   --> $DIR/issue-70935-complex-spans.rs:13:9
            //    |
            // LL |            baz(|| async {
            //    |  ______________-
            //    | |
            //    | |
            // LL | |              foo(tx.clone());
            // LL | |          }).await;
            //    | |          - ^^^^^^ await occurs here, with value maybe used later
            //    | |__________|
            //    |            has type `closure` which is not `Send`
            // note: value is later dropped here
            // LL | |          }).await;
            //    | |                  ^
            //
            span.push_span_label(
                yield_span,
                format!("{await_or_yield} occurs here, with {snippet} maybe used later"),
            );
            span.push_span_label(
                interior_span,
                format!("has type `{target_ty}` which {trait_explanation}"),
            );
            err.span_note(
                span,
                format!("{future_or_coroutine} {trait_explanation} as this value is used across {an_await_or_yield}"),
            );
        };
        match interior_or_upvar_span {
            CoroutineInteriorOrUpvar::Interior(interior_span, interior_extra_info) => {
                if let Some((yield_span, from_awaited_ty)) = interior_extra_info {
                    if let Some(await_span) = from_awaited_ty {
                        // The type causing this obligation is one being awaited at await_span.
                        let mut span = MultiSpan::from_span(await_span);
                        span.push_span_label(
                            await_span,
                            format!(
                                "await occurs here on type `{target_ty}`, which {trait_explanation}"
                            ),
                        );
                        err.span_note(
                            span,
                            format!(
                                "future {trait_explanation} as it awaits another future which {trait_explanation}"
                            ),
                        );
                    } else {
                        // Look at the last interior type to get a span for the `.await`.
                        explain_yield(interior_span, yield_span);
                    }
                }
            }
            CoroutineInteriorOrUpvar::Upvar(upvar_span) => {
                // `Some((ref_ty, is_mut))` if `target_ty` is `&T` or `&mut T` and fails to impl `Send`
                let non_send = match target_ty.kind() {
                    ty::Ref(_, ref_ty, mutability) => match self.evaluate_obligation(obligation) {
                        Ok(eval) if !eval.may_apply() => Some((ref_ty, mutability.is_mut())),
                        _ => None,
                    },
                    _ => None,
                };

                let (span_label, span_note) = match non_send {
                    // if `target_ty` is `&T` or `&mut T` and fails to impl `Send`,
                    // include suggestions to make `T: Sync` so that `&T: Send`,
                    // or to make `T: Send` so that `&mut T: Send`
                    Some((ref_ty, is_mut)) => {
                        let ref_ty_trait = if is_mut { "Send" } else { "Sync" };
                        let ref_kind = if is_mut { "&mut" } else { "&" };
                        (
                            format!(
                                "has type `{target_ty}` which {trait_explanation}, because `{ref_ty}` is not `{ref_ty_trait}`"
                            ),
                            format!(
                                "captured value {trait_explanation} because `{ref_kind}` references cannot be sent unless their referent is `{ref_ty_trait}`"
                            ),
                        )
                    }
                    None => (
                        format!("has type `{target_ty}` which {trait_explanation}"),
                        format!("captured value {trait_explanation}"),
                    ),
                };

                let mut span = MultiSpan::from_span(upvar_span);
                span.push_span_label(upvar_span, span_label);
                err.span_note(span, span_note);
            }
        }

        // Add a note for the item obligation that remains - normally a note pointing to the
        // bound that introduced the obligation (e.g. `T: Send`).
        debug!(?next_code);
        self.note_obligation_cause_code(
            obligation.cause.body_id,
            err,
            obligation.predicate,
            obligation.param_env,
            next_code.unwrap(),
            &mut Vec::new(),
            &mut Default::default(),
        );
    }

    pub(super) fn note_obligation_cause_code<G: EmissionGuarantee, T>(
        &self,
        body_id: LocalDefId,
        err: &mut Diag<'_, G>,
        predicate: T,
        param_env: ty::ParamEnv<'tcx>,
        cause_code: &ObligationCauseCode<'tcx>,
        obligated_types: &mut Vec<Ty<'tcx>>,
        seen_requirements: &mut FxHashSet<DefId>,
    ) where
        T: Upcast<TyCtxt<'tcx>, ty::Predicate<'tcx>>,
    {
        let tcx = self.tcx;
        let predicate = predicate.upcast(tcx);
        let suggest_remove_deref = |err: &mut Diag<'_, G>, expr: &hir::Expr<'_>| {
            if let Some(pred) = predicate.as_trait_clause()
                && tcx.is_lang_item(pred.def_id(), LangItem::Sized)
                && let hir::ExprKind::Unary(hir::UnOp::Deref, inner) = expr.kind
            {
                err.span_suggestion_verbose(
                    expr.span.until(inner.span),
                    "references are always `Sized`, even if they point to unsized data; consider \
                     not dereferencing the expression",
                    String::new(),
                    Applicability::MaybeIncorrect,
                );
            }
        };
        match *cause_code {
            ObligationCauseCode::ExprAssignable
            | ObligationCauseCode::MatchExpressionArm { .. }
            | ObligationCauseCode::Pattern { .. }
            | ObligationCauseCode::IfExpression { .. }
            | ObligationCauseCode::IfExpressionWithNoElse
            | ObligationCauseCode::MainFunctionType
            | ObligationCauseCode::LangFunctionType(_)
            | ObligationCauseCode::IntrinsicType
            | ObligationCauseCode::MethodReceiver
            | ObligationCauseCode::ReturnNoExpression
            | ObligationCauseCode::Misc
            | ObligationCauseCode::WellFormed(..)
            | ObligationCauseCode::MatchImpl(..)
            | ObligationCauseCode::ReturnValue(_)
            | ObligationCauseCode::BlockTailExpression(..)
            | ObligationCauseCode::AwaitableExpr(_)
            | ObligationCauseCode::ForLoopIterator
            | ObligationCauseCode::QuestionMark
            | ObligationCauseCode::CheckAssociatedTypeBounds { .. }
            | ObligationCauseCode::LetElse
            | ObligationCauseCode::BinOp { .. }
            | ObligationCauseCode::AscribeUserTypeProvePredicate(..)
            | ObligationCauseCode::AlwaysApplicableImpl
            | ObligationCauseCode::ConstParam(_)
            | ObligationCauseCode::ReferenceOutlivesReferent(..)
            | ObligationCauseCode::ObjectTypeBound(..) => {}
            ObligationCauseCode::RustCall => {
                if let Some(pred) = predicate.as_trait_clause()
                    && tcx.is_lang_item(pred.def_id(), LangItem::Sized)
                {
                    err.note("argument required to be sized due to `extern \"rust-call\"` ABI");
                }
            }
            ObligationCauseCode::SliceOrArrayElem => {
                err.note("slice and array elements must have `Sized` type");
            }
            ObligationCauseCode::ArrayLen(array_ty) => {
                err.note(format!("the length of array `{array_ty}` must be type `usize`"));
            }
            ObligationCauseCode::TupleElem => {
                err.note("only the last element of a tuple may have a dynamically sized type");
            }
            ObligationCauseCode::DynCompatible(span) => {
                err.multipart_suggestion(
                    "you might have meant to use `Self` to refer to the implementing type",
                    vec![(span, "Self".into())],
                    Applicability::MachineApplicable,
                );
            }
            ObligationCauseCode::WhereClause(item_def_id, span)
            | ObligationCauseCode::WhereClauseInExpr(item_def_id, span, ..)
            | ObligationCauseCode::HostEffectInExpr(item_def_id, span, ..)
                if !span.is_dummy() =>
            {
                if let ObligationCauseCode::WhereClauseInExpr(_, _, hir_id, pos) = &cause_code {
                    if let Node::Expr(expr) = tcx.parent_hir_node(*hir_id)
                        && let hir::ExprKind::Call(_, args) = expr.kind
                        && let Some(expr) = args.get(*pos)
                    {
                        suggest_remove_deref(err, &expr);
                    } else if let Node::Expr(expr) = self.tcx.hir_node(*hir_id)
                        && let hir::ExprKind::MethodCall(_, _, args, _) = expr.kind
                        && let Some(expr) = args.get(*pos)
                    {
                        suggest_remove_deref(err, &expr);
                    }
                }
                let item_name = tcx.def_path_str(item_def_id);
                let short_item_name = with_forced_trimmed_paths!(tcx.def_path_str(item_def_id));
                let mut multispan = MultiSpan::from(span);
                let sm = tcx.sess.source_map();
                if let Some(ident) = tcx.opt_item_ident(item_def_id) {
                    let same_line =
                        match (sm.lookup_line(ident.span.hi()), sm.lookup_line(span.lo())) {
                            (Ok(l), Ok(r)) => l.line == r.line,
                            _ => true,
                        };
                    if ident.span.is_visible(sm) && !ident.span.overlaps(span) && !same_line {
                        multispan.push_span_label(
                            ident.span,
                            format!(
                                "required by a bound in this {}",
                                tcx.def_kind(item_def_id).descr(item_def_id)
                            ),
                        );
                    }
                }
                let mut a = "a";
                let mut this = "this bound";
                let mut note = None;
                let mut help = None;
                if let ty::PredicateKind::Clause(clause) = predicate.kind().skip_binder() {
                    match clause {
                        ty::ClauseKind::Trait(trait_pred) => {
                            let def_id = trait_pred.def_id();
                            let visible_item = if let Some(local) = def_id.as_local() {
                                // Check for local traits being reachable.
                                let vis = &tcx.resolutions(()).effective_visibilities;
                                // Account for non-`pub` traits in the root of the local crate.
                                let is_locally_reachable = tcx.parent(def_id).is_crate_root();
                                vis.is_reachable(local) || is_locally_reachable
                            } else {
                                // Check for foreign traits being reachable.
                                tcx.visible_parent_map(()).get(&def_id).is_some()
                            };
                            if tcx.is_lang_item(def_id, LangItem::Sized) {
                                // Check if this is an implicit bound, even in foreign crates.
                                if tcx
                                    .generics_of(item_def_id)
                                    .own_params
                                    .iter()
                                    .any(|param| tcx.def_span(param.def_id) == span)
                                {
                                    a = "an implicit `Sized`";
                                    this =
                                        "the implicit `Sized` requirement on this type parameter";
                                }
                                if let Some(hir::Node::TraitItem(hir::TraitItem {
                                    generics,
                                    kind: hir::TraitItemKind::Type(bounds, None),
                                    ..
                                })) = tcx.hir_get_if_local(item_def_id)
                                    // Do not suggest relaxing if there is an explicit `Sized` obligation.
                                    && !bounds.iter()
                                        .filter_map(|bound| bound.trait_ref())
                                        .any(|tr| tr.trait_def_id().is_some_and(|def_id| tcx.is_lang_item(def_id, LangItem::Sized)))
                                {
                                    let (span, separator) = if let [.., last] = bounds {
                                        (last.span().shrink_to_hi(), " +")
                                    } else {
                                        (generics.span.shrink_to_hi(), ":")
                                    };
                                    err.span_suggestion_verbose(
                                        span,
                                        "consider relaxing the implicit `Sized` restriction",
                                        format!("{separator} ?Sized"),
                                        Applicability::MachineApplicable,
                                    );
                                }
                            }
                            if let DefKind::Trait = tcx.def_kind(item_def_id)
                                && !visible_item
                            {
                                note = Some(format!(
                                    "`{short_item_name}` is a \"sealed trait\", because to implement it \
                                    you also need to implement `{}`, which is not accessible; this is \
                                    usually done to force you to use one of the provided types that \
                                    already implement it",
                                    with_no_trimmed_paths!(tcx.def_path_str(def_id)),
                                ));
                                let impls_of = tcx.trait_impls_of(def_id);
                                let impls = impls_of
                                    .non_blanket_impls()
                                    .values()
                                    .flatten()
                                    .chain(impls_of.blanket_impls().iter())
                                    .collect::<Vec<_>>();
                                if !impls.is_empty() {
                                    let len = impls.len();
                                    let mut types = impls
                                        .iter()
                                        .map(|t| {
                                            with_no_trimmed_paths!(format!(
                                                "  {}",
                                                tcx.type_of(*t).instantiate_identity(),
                                            ))
                                        })
                                        .collect::<Vec<_>>();
                                    let post = if types.len() > 9 {
                                        types.truncate(8);
                                        format!("\nand {} others", len - 8)
                                    } else {
                                        String::new()
                                    };
                                    help = Some(format!(
                                        "the following type{} implement{} the trait:\n{}{post}",
                                        pluralize!(len),
                                        if len == 1 { "s" } else { "" },
                                        types.join("\n"),
                                    ));
                                }
                            }
                        }
                        ty::ClauseKind::ConstArgHasType(..) => {
                            let descr =
                                format!("required by a const generic parameter in `{item_name}`");
                            if span.is_visible(sm) {
                                let msg = format!(
                                    "required by this const generic parameter in `{short_item_name}`"
                                );
                                multispan.push_span_label(span, msg);
                                err.span_note(multispan, descr);
                            } else {
                                err.span_note(tcx.def_span(item_def_id), descr);
                            }
                            return;
                        }
                        _ => (),
                    }
                }

                // If this is from a format string literal desugaring,
                // we've already said "required by this formatting parameter"
                let is_in_fmt_lit = if let Some(s) = err.span.primary_span() {
                    matches!(s.desugaring_kind(), Some(DesugaringKind::FormatLiteral { .. }))
                } else {
                    false
                };
                if !is_in_fmt_lit {
                    let descr = format!("required by {a} bound in `{item_name}`");
                    if span.is_visible(sm) {
                        let msg = format!("required by {this} in `{short_item_name}`");
                        multispan.push_span_label(span, msg);
                        err.span_note(multispan, descr);
                    } else {
                        err.span_note(tcx.def_span(item_def_id), descr);
                    }
                }
                if let Some(note) = note {
                    err.note(note);
                }
                if let Some(help) = help {
                    err.help(help);
                }
            }
            ObligationCauseCode::WhereClause(..)
            | ObligationCauseCode::WhereClauseInExpr(..)
            | ObligationCauseCode::HostEffectInExpr(..) => {
                // We hold the `DefId` of the item introducing the obligation, but displaying it
                // doesn't add user usable information. It always point at an associated item.
            }
            ObligationCauseCode::OpaqueTypeBound(span, definition_def_id) => {
                err.span_note(span, "required by a bound in an opaque type");
                if let Some(definition_def_id) = definition_def_id
                    // If there are any stalled coroutine obligations, then this
                    // error may be due to that, and not because the body has more
                    // where-clauses.
                    && self.tcx.typeck(definition_def_id).coroutine_stalled_predicates.is_empty()
                {
                    // FIXME(compiler-errors): We could probably point to something
                    // specific here if we tried hard enough...
                    err.span_note(
                        tcx.def_span(definition_def_id),
                        "this definition site has more where clauses than the opaque type",
                    );
                }
            }
            ObligationCauseCode::Coercion { source, target } => {
                let source =
                    tcx.short_string(self.resolve_vars_if_possible(source), err.long_ty_path());
                let target =
                    tcx.short_string(self.resolve_vars_if_possible(target), err.long_ty_path());
                err.note(with_forced_trimmed_paths!(format!(
                    "required for the cast from `{source}` to `{target}`",
                )));
            }
            ObligationCauseCode::RepeatElementCopy { is_constable, elt_span } => {
                err.note(
                    "the `Copy` trait is required because this value will be copied for each element of the array",
                );
                let sm = tcx.sess.source_map();
                if matches!(is_constable, IsConstable::Fn | IsConstable::Ctor)
                    && let Ok(_) = sm.span_to_snippet(elt_span)
                {
                    err.multipart_suggestion(
                        "create an inline `const` block",
                        vec![
                            (elt_span.shrink_to_lo(), "const { ".to_string()),
                            (elt_span.shrink_to_hi(), " }".to_string()),
                        ],
                        Applicability::MachineApplicable,
                    );
                } else {
                    // FIXME: we may suggest array::repeat instead
                    err.help("consider using `core::array::from_fn` to initialize the array");
                    err.help("see https://doc.rust-lang.org/stable/std/array/fn.from_fn.html for more information");
                }
            }
            ObligationCauseCode::VariableType(hir_id) => {
                if let Some(typeck_results) = &self.typeck_results
                    && let Some(ty) = typeck_results.node_type_opt(hir_id)
                    && let ty::Error(_) = ty.kind()
                {
                    err.note(format!(
                        "`{predicate}` isn't satisfied, but the type of this pattern is \
                         `{{type error}}`",
                    ));
                    err.downgrade_to_delayed_bug();
                }
                let mut local = true;
                match tcx.parent_hir_node(hir_id) {
                    Node::LetStmt(hir::LetStmt { ty: Some(ty), .. }) => {
                        err.span_suggestion_verbose(
                            ty.span.shrink_to_lo(),
                            "consider borrowing here",
                            "&",
                            Applicability::MachineApplicable,
                        );
                    }
                    Node::LetStmt(hir::LetStmt {
                        init: Some(hir::Expr { kind: hir::ExprKind::Index(..), span, .. }),
                        ..
                    }) => {
                        // When encountering an assignment of an unsized trait, like
                        // `let x = ""[..];`, provide a suggestion to borrow the initializer in
                        // order to use have a slice instead.
                        err.span_suggestion_verbose(
                            span.shrink_to_lo(),
                            "consider borrowing here",
                            "&",
                            Applicability::MachineApplicable,
                        );
                    }
                    Node::LetStmt(hir::LetStmt { init: Some(expr), .. }) => {
                        // When encountering an assignment of an unsized trait, like `let x = *"";`,
                        // we check if the RHS is a deref operation, to suggest removing it.
                        suggest_remove_deref(err, &expr);
                    }
                    Node::Param(param) => {
                        err.span_suggestion_verbose(
                            param.ty_span.shrink_to_lo(),
                            "function arguments must have a statically known size, borrowed types \
                            always have a known size",
                            "&",
                            Applicability::MachineApplicable,
                        );
                        local = false;
                    }
                    _ => {}
                }
                if local {
                    err.note("all local variables must have a statically known size");
                }
            }
            ObligationCauseCode::SizedArgumentType(hir_id) => {
                let mut ty = None;
                let borrowed_msg = "function arguments must have a statically known size, borrowed \
                                    types always have a known size";
                if let Some(hir_id) = hir_id
                    && let hir::Node::Param(param) = self.tcx.hir_node(hir_id)
                    && let Some(decl) = self.tcx.parent_hir_node(hir_id).fn_decl()
                    && let Some(t) = decl.inputs.iter().find(|t| param.ty_span.contains(t.span))
                {
                    // We use `contains` because the type might be surrounded by parentheses,
                    // which makes `ty_span` and `t.span` disagree with each other, but one
                    // fully contains the other: `foo: (dyn Foo + Bar)`
                    //                                 ^-------------^
                    //                                 ||
                    //                                 |t.span
                    //                                 param._ty_span
                    ty = Some(t);
                } else if let Some(hir_id) = hir_id
                    && let hir::Node::Ty(t) = self.tcx.hir_node(hir_id)
                {
                    ty = Some(t);
                }
                if let Some(ty) = ty {
                    match ty.kind {
                        hir::TyKind::TraitObject(traits, _) => {
                            let (span, kw) = match traits {
                                [first, ..] if first.span.lo() == ty.span.lo() => {
                                    // Missing `dyn` in front of trait object.
                                    (ty.span.shrink_to_lo(), "dyn ")
                                }
                                [first, ..] => (ty.span.until(first.span), ""),
                                [] => span_bug!(ty.span, "trait object with no traits: {ty:?}"),
                            };
                            let needs_parens = traits.len() != 1;
                            // Don't recommend impl Trait as a closure argument
                            if let Some(hir_id) = hir_id
                                && matches!(
                                    self.tcx.parent_hir_node(hir_id),
                                    hir::Node::Item(hir::Item {
                                        kind: hir::ItemKind::Fn { .. },
                                        ..
                                    })
                                )
                            {
                                err.span_suggestion_verbose(
                                    span,
                                    "you can use `impl Trait` as the argument type",
                                    "impl ",
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            let sugg = if !needs_parens {
                                vec![(span.shrink_to_lo(), format!("&{kw}"))]
                            } else {
                                vec![
                                    (span.shrink_to_lo(), format!("&({kw}")),
                                    (ty.span.shrink_to_hi(), ")".to_string()),
                                ]
                            };
                            err.multipart_suggestion_verbose(
                                borrowed_msg,
                                sugg,
                                Applicability::MachineApplicable,
                            );
                        }
                        hir::TyKind::Slice(_ty) => {
                            err.span_suggestion_verbose(
                                ty.span.shrink_to_lo(),
                                "function arguments must have a statically known size, borrowed \
                                 slices always have a known size",
                                "&",
                                Applicability::MachineApplicable,
                            );
                        }
                        hir::TyKind::Path(_) => {
                            err.span_suggestion_verbose(
                                ty.span.shrink_to_lo(),
                                borrowed_msg,
                                "&",
                                Applicability::MachineApplicable,
                            );
                        }
                        _ => {}
                    }
                } else {
                    err.note("all function arguments must have a statically known size");
                }
                if tcx.sess.opts.unstable_features.is_nightly_build()
                    && !tcx.features().unsized_fn_params()
                {
                    err.help("unsized fn params are gated as an unstable feature");
                }
            }
            ObligationCauseCode::SizedReturnType | ObligationCauseCode::SizedCallReturnType => {
                err.note("the return type of a function must have a statically known size");
            }
            ObligationCauseCode::SizedYieldType => {
                err.note("the yield type of a coroutine must have a statically known size");
            }
            ObligationCauseCode::AssignmentLhsSized => {
                err.note("the left-hand-side of an assignment must have a statically known size");
            }
            ObligationCauseCode::TupleInitializerSized => {
                err.note("tuples must have a statically known size to be initialized");
            }
            ObligationCauseCode::StructInitializerSized => {
                err.note("structs must have a statically known size to be initialized");
            }
            ObligationCauseCode::FieldSized { adt_kind: ref item, last, span } => {
                match *item {
                    AdtKind::Struct => {
                        if last {
                            err.note(
                                "the last field of a packed struct may only have a \
                                dynamically sized type if it does not need drop to be run",
                            );
                        } else {
                            err.note(
                                "only the last field of a struct may have a dynamically sized type",
                            );
                        }
                    }
                    AdtKind::Union => {
                        err.note("no field of a union may have a dynamically sized type");
                    }
                    AdtKind::Enum => {
                        err.note("no field of an enum variant may have a dynamically sized type");
                    }
                }
                err.help("change the field's type to have a statically known size");
                err.span_suggestion_verbose(
                    span.shrink_to_lo(),
                    "borrowed types always have a statically known size",
                    "&",
                    Applicability::MachineApplicable,
                );
                err.multipart_suggestion_verbose(
                    "the `Box` type always has a statically known size and allocates its contents \
                     in the heap",
                    vec![
                        (span.shrink_to_lo(), "Box<".to_string()),
                        (span.shrink_to_hi(), ">".to_string()),
                    ],
                    Applicability::MachineApplicable,
                );
            }
            ObligationCauseCode::SizedConstOrStatic => {
                err.note("statics and constants must have a statically known size");
            }
            ObligationCauseCode::InlineAsmSized => {
                err.note("all inline asm arguments must have a statically known size");
            }
            ObligationCauseCode::SizedClosureCapture(closure_def_id) => {
                err.note(
                    "all values captured by value by a closure must have a statically known size",
                );
                let hir::ExprKind::Closure(closure) =
                    tcx.hir_node_by_def_id(closure_def_id).expect_expr().kind
                else {
                    bug!("expected closure in SizedClosureCapture obligation");
                };
                if let hir::CaptureBy::Value { .. } = closure.capture_clause
                    && let Some(span) = closure.fn_arg_span
                {
                    err.span_label(span, "this closure captures all values by move");
                }
            }
            ObligationCauseCode::SizedCoroutineInterior(coroutine_def_id) => {
                let what = match tcx.coroutine_kind(coroutine_def_id) {
                    None
                    | Some(hir::CoroutineKind::Coroutine(_))
                    | Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen, _)) => {
                        "yield"
                    }
                    Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _)) => {
                        "await"
                    }
                    Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen, _)) => {
                        "yield`/`await"
                    }
                };
                err.note(format!(
                    "all values live across `{what}` must have a statically known size"
                ));
            }
            ObligationCauseCode::SharedStatic => {
                err.note("shared static variables must have a type that implements `Sync`");
            }
            ObligationCauseCode::BuiltinDerived(ref data) => {
                let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_pred);
                let ty = parent_trait_ref.skip_binder().self_ty();
                if parent_trait_ref.references_error() {
                    // NOTE(eddyb) this was `.cancel()`, but `err`
                    // is borrowed, so we can't fully defuse it.
                    err.downgrade_to_delayed_bug();
                    return;
                }

                // If the obligation for a tuple is set directly by a Coroutine or Closure,
                // then the tuple must be the one containing capture types.
                let is_upvar_tys_infer_tuple = if !matches!(ty.kind(), ty::Tuple(..)) {
                    false
                } else if let ObligationCauseCode::BuiltinDerived(data) = &*data.parent_code {
                    let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_pred);
                    let nested_ty = parent_trait_ref.skip_binder().self_ty();
                    matches!(nested_ty.kind(), ty::Coroutine(..))
                        || matches!(nested_ty.kind(), ty::Closure(..))
                } else {
                    false
                };

                let is_builtin_async_fn_trait =
                    tcx.async_fn_trait_kind_from_def_id(data.parent_trait_pred.def_id()).is_some();

                if !is_upvar_tys_infer_tuple && !is_builtin_async_fn_trait {
                    let ty_str = tcx.short_string(ty, err.long_ty_path());
                    let msg = format!("required because it appears within the type `{ty_str}`");
                    match ty.kind() {
                        ty::Adt(def, _) => match tcx.opt_item_ident(def.did()) {
                            Some(ident) => {
                                err.span_note(ident.span, msg);
                            }
                            None => {
                                err.note(msg);
                            }
                        },
                        ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }) => {
                            // If the previous type is async fn, this is the future generated by the body of an async function.
                            // Avoid printing it twice (it was already printed in the `ty::Coroutine` arm below).
                            let is_future = tcx.ty_is_opaque_future(ty);
                            debug!(
                                ?obligated_types,
                                ?is_future,
                                "note_obligation_cause_code: check for async fn"
                            );
                            if is_future
                                && obligated_types.last().is_some_and(|ty| match ty.kind() {
                                    ty::Coroutine(last_def_id, ..) => {
                                        tcx.coroutine_is_async(*last_def_id)
                                    }
                                    _ => false,
                                })
                            {
                                // See comment above; skip printing twice.
                            } else {
                                err.span_note(tcx.def_span(def_id), msg);
                            }
                        }
                        ty::Coroutine(def_id, _) => {
                            let sp = tcx.def_span(def_id);

                            // Special-case this to say "async block" instead of `[static coroutine]`.
                            let kind = tcx.coroutine_kind(def_id).unwrap();
                            err.span_note(
                                sp,
                                with_forced_trimmed_paths!(format!(
                                    "required because it's used within this {kind:#}",
                                )),
                            );
                        }
                        ty::CoroutineWitness(..) => {
                            // Skip printing coroutine-witnesses, since we'll drill into
                            // the bad field in another derived obligation cause.
                        }
                        ty::Closure(def_id, _) | ty::CoroutineClosure(def_id, _) => {
                            err.span_note(
                                tcx.def_span(def_id),
                                "required because it's used within this closure",
                            );
                        }
                        ty::Str => {
                            err.note("`str` is considered to contain a `[u8]` slice for auto trait purposes");
                        }
                        _ => {
                            err.note(msg);
                        }
                    };
                }

                obligated_types.push(ty);

                let parent_predicate = parent_trait_ref;
                if !self.is_recursive_obligation(obligated_types, &data.parent_code) {
                    // #74711: avoid a stack overflow
                    ensure_sufficient_stack(|| {
                        self.note_obligation_cause_code(
                            body_id,
                            err,
                            parent_predicate,
                            param_env,
                            &data.parent_code,
                            obligated_types,
                            seen_requirements,
                        )
                    });
                } else {
                    ensure_sufficient_stack(|| {
                        self.note_obligation_cause_code(
                            body_id,
                            err,
                            parent_predicate,
                            param_env,
                            cause_code.peel_derives(),
                            obligated_types,
                            seen_requirements,
                        )
                    });
                }
            }
            ObligationCauseCode::ImplDerived(ref data) => {
                let mut parent_trait_pred =
                    self.resolve_vars_if_possible(data.derived.parent_trait_pred);
                let parent_def_id = parent_trait_pred.def_id();
                if tcx.is_diagnostic_item(sym::FromResidual, parent_def_id)
                    && !tcx.features().enabled(sym::try_trait_v2)
                {
                    // If `#![feature(try_trait_v2)]` is not enabled, then there's no point on
                    // talking about `FromResidual<Result<A, B>>`, as the end user has nothing they
                    // can do about it. As far as they are concerned, `?` is compiler magic.
                    return;
                }
                let self_ty_str =
                    tcx.short_string(parent_trait_pred.skip_binder().self_ty(), err.long_ty_path());
                let trait_name = parent_trait_pred.print_modifiers_and_trait_path().to_string();
                let msg = format!("required for `{self_ty_str}` to implement `{trait_name}`");
                let mut is_auto_trait = false;
                match tcx.hir_get_if_local(data.impl_or_alias_def_id) {
                    Some(Node::Item(hir::Item {
                        kind: hir::ItemKind::Trait(is_auto, _, ident, ..),
                        ..
                    })) => {
                        // FIXME: we should do something else so that it works even on crate foreign
                        // auto traits.
                        is_auto_trait = matches!(is_auto, hir::IsAuto::Yes);
                        err.span_note(ident.span, msg);
                    }
                    Some(Node::Item(hir::Item {
                        kind: hir::ItemKind::Impl(hir::Impl { of_trait, self_ty, generics, .. }),
                        ..
                    })) => {
                        let mut spans = Vec::with_capacity(2);
                        if let Some(trait_ref) = of_trait {
                            spans.push(trait_ref.path.span);
                        }
                        spans.push(self_ty.span);
                        let mut spans: MultiSpan = spans.into();
                        if matches!(
                            self_ty.span.ctxt().outer_expn_data().kind,
                            ExpnKind::Macro(MacroKind::Derive, _)
                        ) || matches!(
                            of_trait.as_ref().map(|t| t.path.span.ctxt().outer_expn_data().kind),
                            Some(ExpnKind::Macro(MacroKind::Derive, _))
                        ) {
                            spans.push_span_label(
                                data.span,
                                "unsatisfied trait bound introduced in this `derive` macro",
                            );
                        } else if !data.span.is_dummy() && !data.span.overlaps(self_ty.span) {
                            spans.push_span_label(
                                data.span,
                                "unsatisfied trait bound introduced here",
                            );
                        }
                        err.span_note(spans, msg);
                        point_at_assoc_type_restriction(
                            tcx,
                            err,
                            &self_ty_str,
                            &trait_name,
                            predicate,
                            &generics,
                            &data,
                        );
                    }
                    _ => {
                        err.note(msg);
                    }
                };

                let mut parent_predicate = parent_trait_pred;
                let mut data = &data.derived;
                let mut count = 0;
                seen_requirements.insert(parent_def_id);
                if is_auto_trait {
                    // We don't want to point at the ADT saying "required because it appears within
                    // the type `X`", like we would otherwise do in test `supertrait-auto-trait.rs`.
                    while let ObligationCauseCode::BuiltinDerived(derived) = &*data.parent_code {
                        let child_trait_ref =
                            self.resolve_vars_if_possible(derived.parent_trait_pred);
                        let child_def_id = child_trait_ref.def_id();
                        if seen_requirements.insert(child_def_id) {
                            break;
                        }
                        data = derived;
                        parent_predicate = child_trait_ref.upcast(tcx);
                        parent_trait_pred = child_trait_ref;
                    }
                }
                while let ObligationCauseCode::ImplDerived(child) = &*data.parent_code {
                    // Skip redundant recursive obligation notes. See `ui/issue-20413.rs`.
                    let child_trait_pred =
                        self.resolve_vars_if_possible(child.derived.parent_trait_pred);
                    let child_def_id = child_trait_pred.def_id();
                    if seen_requirements.insert(child_def_id) {
                        break;
                    }
                    count += 1;
                    data = &child.derived;
                    parent_predicate = child_trait_pred.upcast(tcx);
                    parent_trait_pred = child_trait_pred;
                }
                if count > 0 {
                    err.note(format!(
                        "{} redundant requirement{} hidden",
                        count,
                        pluralize!(count)
                    ));
                    let self_ty = tcx.short_string(
                        parent_trait_pred.skip_binder().self_ty(),
                        err.long_ty_path(),
                    );
                    err.note(format!(
                        "required for `{self_ty}` to implement `{}`",
                        parent_trait_pred.print_modifiers_and_trait_path()
                    ));
                }
                // #74711: avoid a stack overflow
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        body_id,
                        err,
                        parent_predicate,
                        param_env,
                        &data.parent_code,
                        obligated_types,
                        seen_requirements,
                    )
                });
            }
            ObligationCauseCode::ImplDerivedHost(ref data) => {
                let self_ty =
                    self.resolve_vars_if_possible(data.derived.parent_host_pred.self_ty());
                let msg = format!(
                    "required for `{self_ty}` to implement `{} {}`",
                    data.derived.parent_host_pred.skip_binder().constness,
                    data.derived
                        .parent_host_pred
                        .map_bound(|pred| pred.trait_ref)
                        .print_only_trait_path(),
                );
                match tcx.hir_get_if_local(data.impl_def_id) {
                    Some(Node::Item(hir::Item {
                        kind: hir::ItemKind::Impl(hir::Impl { of_trait, self_ty, .. }),
                        ..
                    })) => {
                        let mut spans = vec![self_ty.span];
                        spans.extend(of_trait.as_ref().map(|t| t.path.span));
                        let mut spans: MultiSpan = spans.into();
                        spans.push_span_label(data.span, "unsatisfied trait bound introduced here");
                        err.span_note(spans, msg);
                    }
                    _ => {
                        err.note(msg);
                    }
                }
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        body_id,
                        err,
                        data.derived.parent_host_pred,
                        param_env,
                        &data.derived.parent_code,
                        obligated_types,
                        seen_requirements,
                    )
                });
            }
            ObligationCauseCode::BuiltinDerivedHost(ref data) => {
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        body_id,
                        err,
                        data.parent_host_pred,
                        param_env,
                        &data.parent_code,
                        obligated_types,
                        seen_requirements,
                    )
                });
            }
            ObligationCauseCode::WellFormedDerived(ref data) => {
                let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_pred);
                let parent_predicate = parent_trait_ref;
                // #74711: avoid a stack overflow
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        body_id,
                        err,
                        parent_predicate,
                        param_env,
                        &data.parent_code,
                        obligated_types,
                        seen_requirements,
                    )
                });
            }
            ObligationCauseCode::TypeAlias(ref nested, span, def_id) => {
                // #74711: avoid a stack overflow
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        body_id,
                        err,
                        predicate,
                        param_env,
                        nested,
                        obligated_types,
                        seen_requirements,
                    )
                });
                let mut multispan = MultiSpan::from(span);
                multispan.push_span_label(span, "required by this bound");
                err.span_note(
                    multispan,
                    format!("required by a bound on the type alias `{}`", tcx.item_name(def_id)),
                );
            }
            ObligationCauseCode::FunctionArg {
                arg_hir_id, call_hir_id, ref parent_code, ..
            } => {
                self.note_function_argument_obligation(
                    body_id,
                    err,
                    arg_hir_id,
                    parent_code,
                    param_env,
                    predicate,
                    call_hir_id,
                );
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        body_id,
                        err,
                        predicate,
                        param_env,
                        parent_code,
                        obligated_types,
                        seen_requirements,
                    )
                });
            }
            // Suppress `compare_type_predicate_entailment` errors for RPITITs, since they
            // should be implied by the parent method.
            ObligationCauseCode::CompareImplItem { trait_item_def_id, .. }
                if tcx.is_impl_trait_in_trait(trait_item_def_id) => {}
            ObligationCauseCode::CompareImplItem { trait_item_def_id, kind, .. } => {
                let item_name = tcx.item_name(trait_item_def_id);
                let msg = format!(
                    "the requirement `{predicate}` appears on the `impl`'s {kind} \
                     `{item_name}` but not on the corresponding trait's {kind}",
                );
                let sp = tcx
                    .opt_item_ident(trait_item_def_id)
                    .map(|i| i.span)
                    .unwrap_or_else(|| tcx.def_span(trait_item_def_id));
                let mut assoc_span: MultiSpan = sp.into();
                assoc_span.push_span_label(
                    sp,
                    format!("this trait's {kind} doesn't have the requirement `{predicate}`"),
                );
                if let Some(ident) = tcx
                    .opt_associated_item(trait_item_def_id)
                    .and_then(|i| tcx.opt_item_ident(i.container_id(tcx)))
                {
                    assoc_span.push_span_label(ident.span, "in this trait");
                }
                err.span_note(assoc_span, msg);
            }
            ObligationCauseCode::TrivialBound => {
                err.help("see issue #48214");
                tcx.disabled_nightly_features(err, [(String::new(), sym::trivial_bounds)]);
            }
            ObligationCauseCode::OpaqueReturnType(expr_info) => {
                let (expr_ty, expr) = if let Some((expr_ty, hir_id)) = expr_info {
                    let expr_ty = tcx.short_string(expr_ty, err.long_ty_path());
                    let expr = tcx.hir_expect_expr(hir_id);
                    (expr_ty, expr)
                } else if let Some(body_id) = tcx.hir_node_by_def_id(body_id).body_id()
                    && let body = tcx.hir_body(body_id)
                    && let hir::ExprKind::Block(block, _) = body.value.kind
                    && let Some(expr) = block.expr
                    && let Some(expr_ty) = self
                        .typeck_results
                        .as_ref()
                        .and_then(|typeck| typeck.node_type_opt(expr.hir_id))
                    && let Some(pred) = predicate.as_clause()
                    && let ty::ClauseKind::Trait(pred) = pred.kind().skip_binder()
                    && self.can_eq(param_env, pred.self_ty(), expr_ty)
                {
                    let expr_ty = tcx.short_string(expr_ty, err.long_ty_path());
                    (expr_ty, expr)
                } else {
                    return;
                };
                err.span_label(
                    expr.span,
                    with_forced_trimmed_paths!(format!(
                        "return type was inferred to be `{expr_ty}` here",
                    )),
                );
                suggest_remove_deref(err, &expr);
            }
        }
    }

    #[instrument(
        level = "debug", skip(self, err), fields(trait_pred.self_ty = ?trait_pred.self_ty())
    )]
    pub(super) fn suggest_await_before_try(
        &self,
        err: &mut Diag<'_>,
        obligation: &PredicateObligation<'tcx>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        span: Span,
    ) {
        let future_trait = self.tcx.require_lang_item(LangItem::Future, span);

        let self_ty = self.resolve_vars_if_possible(trait_pred.self_ty());
        let impls_future = self.type_implements_trait(
            future_trait,
            [self.tcx.instantiate_bound_regions_with_erased(self_ty)],
            obligation.param_env,
        );
        if !impls_future.must_apply_modulo_regions() {
            return;
        }

        let item_def_id = self.tcx.associated_item_def_ids(future_trait)[0];
        // `<T as Future>::Output`
        let projection_ty = trait_pred.map_bound(|trait_pred| {
            Ty::new_projection(
                self.tcx,
                item_def_id,
                // Future::Output has no args
                [trait_pred.self_ty()],
            )
        });
        let InferOk { value: projection_ty, .. } =
            self.at(&obligation.cause, obligation.param_env).normalize(projection_ty);

        debug!(
            normalized_projection_type = ?self.resolve_vars_if_possible(projection_ty)
        );
        let try_obligation = self.mk_trait_obligation_with_new_self_ty(
            obligation.param_env,
            trait_pred.map_bound(|trait_pred| (trait_pred, projection_ty.skip_binder())),
        );
        debug!(try_trait_obligation = ?try_obligation);
        if self.predicate_may_hold(&try_obligation)
            && let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span)
            && snippet.ends_with('?')
        {
            match self.tcx.coroutine_kind(obligation.cause.body_id) {
                Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _)) => {
                    err.span_suggestion_verbose(
                        span.with_hi(span.hi() - BytePos(1)).shrink_to_hi(),
                        "consider `await`ing on the `Future`",
                        ".await",
                        Applicability::MaybeIncorrect,
                    );
                }
                _ => {
                    let mut span: MultiSpan = span.with_lo(span.hi() - BytePos(1)).into();
                    span.push_span_label(
                        self.tcx.def_span(obligation.cause.body_id),
                        "this is not `async`",
                    );
                    err.span_note(
                        span,
                        "this implements `Future` and its output type supports \
                        `?`, but the future cannot be awaited in a synchronous function",
                    );
                }
            }
        }
    }

    pub(super) fn suggest_floating_point_literal(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) {
        let rhs_span = match obligation.cause.code() {
            ObligationCauseCode::BinOp { rhs_span: Some(span), rhs_is_lit, .. } if *rhs_is_lit => {
                span
            }
            _ => return,
        };
        if let ty::Float(_) = trait_pred.skip_binder().self_ty().kind()
            && let ty::Infer(InferTy::IntVar(_)) =
                trait_pred.skip_binder().trait_ref.args.type_at(1).kind()
        {
            err.span_suggestion_verbose(
                rhs_span.shrink_to_hi(),
                "consider using a floating-point literal by writing it with `.0`",
                ".0",
                Applicability::MaybeIncorrect,
            );
        }
    }

    pub fn suggest_derive(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) {
        if trait_pred.polarity() == ty::PredicatePolarity::Negative {
            return;
        }
        let Some(diagnostic_name) = self.tcx.get_diagnostic_name(trait_pred.def_id()) else {
            return;
        };
        let (adt, args) = match trait_pred.skip_binder().self_ty().kind() {
            ty::Adt(adt, args) if adt.did().is_local() => (adt, args),
            _ => return,
        };
        let can_derive = {
            let is_derivable_trait = match diagnostic_name {
                sym::Default => !adt.is_enum(),
                sym::PartialEq | sym::PartialOrd => {
                    let rhs_ty = trait_pred.skip_binder().trait_ref.args.type_at(1);
                    trait_pred.skip_binder().self_ty() == rhs_ty
                }
                sym::Eq | sym::Ord | sym::Clone | sym::Copy | sym::Hash | sym::Debug => true,
                _ => false,
            };
            is_derivable_trait &&
                // Ensure all fields impl the trait.
                adt.all_fields().all(|field| {
                    let field_ty = ty::GenericArg::from(field.ty(self.tcx, args));
                    let trait_args = match diagnostic_name {
                        sym::PartialEq | sym::PartialOrd => {
                            Some(field_ty)
                        }
                        _ => None,
                    };
                    let trait_pred = trait_pred.map_bound_ref(|tr| ty::TraitPredicate {
                        trait_ref: ty::TraitRef::new(self.tcx,
                            trait_pred.def_id(),
                            [field_ty].into_iter().chain(trait_args),
                        ),
                        ..*tr
                    });
                    let field_obl = Obligation::new(
                        self.tcx,
                        obligation.cause.clone(),
                        obligation.param_env,
                        trait_pred,
                    );
                    self.predicate_must_hold_modulo_regions(&field_obl)
                })
        };
        if can_derive {
            err.span_suggestion_verbose(
                self.tcx.def_span(adt.did()).shrink_to_lo(),
                format!(
                    "consider annotating `{}` with `#[derive({})]`",
                    trait_pred.skip_binder().self_ty(),
                    diagnostic_name,
                ),
                // FIXME(const_trait_impl) derive_const as suggestion?
                format!("#[derive({diagnostic_name})]\n"),
                Applicability::MaybeIncorrect,
            );
        }
    }

    pub(super) fn suggest_dereferencing_index(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) {
        if let ObligationCauseCode::ImplDerived(_) = obligation.cause.code()
            && self
                .tcx
                .is_diagnostic_item(sym::SliceIndex, trait_pred.skip_binder().trait_ref.def_id)
            && let ty::Slice(_) = trait_pred.skip_binder().trait_ref.args.type_at(1).kind()
            && let ty::Ref(_, inner_ty, _) = trait_pred.skip_binder().self_ty().kind()
            && let ty::Uint(ty::UintTy::Usize) = inner_ty.kind()
        {
            err.span_suggestion_verbose(
                obligation.cause.span.shrink_to_lo(),
                "dereference this index",
                '*',
                Applicability::MachineApplicable,
            );
        }
    }

    fn note_function_argument_obligation<G: EmissionGuarantee>(
        &self,
        body_id: LocalDefId,
        err: &mut Diag<'_, G>,
        arg_hir_id: HirId,
        parent_code: &ObligationCauseCode<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        failed_pred: ty::Predicate<'tcx>,
        call_hir_id: HirId,
    ) {
        let tcx = self.tcx;
        if let Node::Expr(expr) = tcx.hir_node(arg_hir_id)
            && let Some(typeck_results) = &self.typeck_results
        {
            if let hir::Expr { kind: hir::ExprKind::MethodCall(_, rcvr, _, _), .. } = expr
                && let Some(ty) = typeck_results.node_type_opt(rcvr.hir_id)
                && let Some(failed_pred) = failed_pred.as_trait_clause()
                && let pred = failed_pred.map_bound(|pred| pred.with_self_ty(tcx, ty))
                && self.predicate_must_hold_modulo_regions(&Obligation::misc(
                    tcx, expr.span, body_id, param_env, pred,
                ))
                && expr.span.hi() != rcvr.span.hi()
            {
                err.span_suggestion_verbose(
                    expr.span.with_lo(rcvr.span.hi()),
                    format!(
                        "consider removing this method call, as the receiver has type `{ty}` and \
                         `{pred}` trivially holds",
                    ),
                    "",
                    Applicability::MaybeIncorrect,
                );
            }
            if let hir::Expr { kind: hir::ExprKind::Block(block, _), .. } = expr {
                let inner_expr = expr.peel_blocks();
                let ty = typeck_results
                    .expr_ty_adjusted_opt(inner_expr)
                    .unwrap_or(Ty::new_misc_error(tcx));
                let span = inner_expr.span;
                if Some(span) != err.span.primary_span()
                    && !span.in_external_macro(tcx.sess.source_map())
                {
                    err.span_label(
                        span,
                        if ty.references_error() {
                            String::new()
                        } else {
                            let ty = with_forced_trimmed_paths!(self.ty_to_string(ty));
                            format!("this tail expression is of type `{ty}`")
                        },
                    );
                    if let ty::PredicateKind::Clause(clause) = failed_pred.kind().skip_binder()
                        && let ty::ClauseKind::Trait(pred) = clause
                        && tcx.fn_trait_kind_from_def_id(pred.def_id()).is_some()
                    {
                        if let [stmt, ..] = block.stmts
                            && let hir::StmtKind::Semi(value) = stmt.kind
                            && let hir::ExprKind::Closure(hir::Closure {
                                body, fn_decl_span, ..
                            }) = value.kind
                            && let body = tcx.hir_body(*body)
                            && !matches!(body.value.kind, hir::ExprKind::Block(..))
                        {
                            // Check if the failed predicate was an expectation of a closure type
                            // and if there might have been a `{ |args|` typo instead of `|args| {`.
                            err.multipart_suggestion(
                                "you might have meant to open the closure body instead of placing \
                                 a closure within a block",
                                vec![
                                    (expr.span.with_hi(value.span.lo()), String::new()),
                                    (fn_decl_span.shrink_to_hi(), " {".to_string()),
                                ],
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            // Maybe the bare block was meant to be a closure.
                            err.span_suggestion_verbose(
                                expr.span.shrink_to_lo(),
                                "you might have meant to create the closure instead of a block",
                                format!(
                                    "|{}| ",
                                    (0..pred.trait_ref.args.len() - 1)
                                        .map(|_| "_")
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                ),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                }
            }

            // FIXME: visit the ty to see if there's any closure involved, and if there is,
            // check whether its evaluated return type is the same as the one corresponding
            // to an associated type (as seen from `trait_pred`) in the predicate. Like in
            // trait_pred `S: Sum<<Self as Iterator>::Item>` and predicate `i32: Sum<&()>`
            let mut type_diffs = vec![];
            if let ObligationCauseCode::WhereClauseInExpr(def_id, _, _, idx) = parent_code
                && let Some(node_args) = typeck_results.node_args_opt(call_hir_id)
                && let where_clauses =
                    self.tcx.predicates_of(def_id).instantiate(self.tcx, node_args)
                && let Some(where_pred) = where_clauses.predicates.get(*idx)
            {
                if let Some(where_pred) = where_pred.as_trait_clause()
                    && let Some(failed_pred) = failed_pred.as_trait_clause()
                    && where_pred.def_id() == failed_pred.def_id()
                {
                    self.enter_forall(where_pred, |where_pred| {
                        let failed_pred = self.instantiate_binder_with_fresh_vars(
                            expr.span,
                            BoundRegionConversionTime::FnCall,
                            failed_pred,
                        );

                        let zipped =
                            iter::zip(where_pred.trait_ref.args, failed_pred.trait_ref.args);
                        for (expected, actual) in zipped {
                            self.probe(|_| {
                                match self
                                    .at(&ObligationCause::misc(expr.span, body_id), param_env)
                                    // Doesn't actually matter if we define opaque types here, this is just used for
                                    // diagnostics, and the result is never kept around.
                                    .eq(DefineOpaqueTypes::Yes, expected, actual)
                                {
                                    Ok(_) => (), // We ignore nested obligations here for now.
                                    Err(err) => type_diffs.push(err),
                                }
                            })
                        }
                    })
                } else if let Some(where_pred) = where_pred.as_projection_clause()
                    && let Some(failed_pred) = failed_pred.as_projection_clause()
                    && let Some(found) = failed_pred.skip_binder().term.as_type()
                {
                    type_diffs = vec![TypeError::Sorts(ty::error::ExpectedFound {
                        expected: where_pred
                            .skip_binder()
                            .projection_term
                            .expect_ty(self.tcx)
                            .to_ty(self.tcx),
                        found,
                    })];
                }
            }
            if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = expr.kind
                && let hir::Path { res: Res::Local(hir_id), .. } = path
                && let hir::Node::Pat(binding) = self.tcx.hir_node(*hir_id)
                && let hir::Node::LetStmt(local) = self.tcx.parent_hir_node(binding.hir_id)
                && let Some(binding_expr) = local.init
            {
                // If the expression we're calling on is a binding, we want to point at the
                // `let` when talking about the type. Otherwise we'll point at every part
                // of the method chain with the type.
                self.point_at_chain(binding_expr, typeck_results, type_diffs, param_env, err);
            } else {
                self.point_at_chain(expr, typeck_results, type_diffs, param_env, err);
            }
        }
        let call_node = tcx.hir_node(call_hir_id);
        if let Node::Expr(hir::Expr { kind: hir::ExprKind::MethodCall(path, rcvr, ..), .. }) =
            call_node
        {
            if Some(rcvr.span) == err.span.primary_span() {
                err.replace_span_with(path.ident.span, true);
            }
        }

        if let Node::Expr(expr) = call_node {
            if let hir::ExprKind::Call(hir::Expr { span, .. }, _)
            | hir::ExprKind::MethodCall(
                hir::PathSegment { ident: Ident { span, .. }, .. },
                ..,
            ) = expr.kind
            {
                if Some(*span) != err.span.primary_span() {
                    let msg = if span.is_desugaring(DesugaringKind::FormatLiteral { source: true })
                    {
                        "required by this formatting parameter"
                    } else if span.is_desugaring(DesugaringKind::FormatLiteral { source: false }) {
                        "required by a formatting parameter in this expression"
                    } else {
                        "required by a bound introduced by this call"
                    };
                    err.span_label(*span, msg);
                }
            }

            if let hir::ExprKind::MethodCall(_, expr, ..) = expr.kind {
                self.suggest_option_method_if_applicable(failed_pred, param_env, err, expr);
            }
        }
    }

    fn suggest_option_method_if_applicable<G: EmissionGuarantee>(
        &self,
        failed_pred: ty::Predicate<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        err: &mut Diag<'_, G>,
        expr: &hir::Expr<'_>,
    ) {
        let tcx = self.tcx;
        let infcx = self.infcx;
        let Some(typeck_results) = self.typeck_results.as_ref() else { return };

        // Make sure we're dealing with the `Option` type.
        let Some(option_ty_adt) = typeck_results.expr_ty_adjusted(expr).ty_adt_def() else {
            return;
        };
        if !tcx.is_diagnostic_item(sym::Option, option_ty_adt.did()) {
            return;
        }

        // Given the predicate `fn(&T): FnOnce<(U,)>`, extract `fn(&T)` and `(U,)`,
        // then suggest `Option::as_deref(_mut)` if `U` can deref to `T`
        if let ty::PredicateKind::Clause(ty::ClauseKind::Trait(ty::TraitPredicate { trait_ref, .. }))
            = failed_pred.kind().skip_binder()
            && tcx.is_fn_trait(trait_ref.def_id)
            && let [self_ty, found_ty] = trait_ref.args.as_slice()
            && let Some(fn_ty) = self_ty.as_type().filter(|ty| ty.is_fn())
            && let fn_sig @ ty::FnSig {
                abi: ExternAbi::Rust,
                c_variadic: false,
                safety: hir::Safety::Safe,
                ..
            } = fn_ty.fn_sig(tcx).skip_binder()

            // Extract first param of fn sig with peeled refs, e.g. `fn(&T)` -> `T`
            && let Some(&ty::Ref(_, target_ty, needs_mut)) = fn_sig.inputs().first().map(|t| t.kind())
            && !target_ty.has_escaping_bound_vars()

            // Extract first tuple element out of fn trait, e.g. `FnOnce<(U,)>` -> `U`
            && let Some(ty::Tuple(tys)) = found_ty.as_type().map(Ty::kind)
            && let &[found_ty] = tys.as_slice()
            && !found_ty.has_escaping_bound_vars()

            // Extract `<U as Deref>::Target` assoc type and check that it is `T`
            && let Some(deref_target_did) = tcx.lang_items().deref_target()
            && let projection = Ty::new_projection_from_args(tcx,deref_target_did, tcx.mk_args(&[ty::GenericArg::from(found_ty)]))
            && let InferOk { value: deref_target, obligations } = infcx.at(&ObligationCause::dummy(), param_env).normalize(projection)
            && obligations.iter().all(|obligation| infcx.predicate_must_hold_modulo_regions(obligation))
            && infcx.can_eq(param_env, deref_target, target_ty)
        {
            let help = if let hir::Mutability::Mut = needs_mut
                && let Some(deref_mut_did) = tcx.lang_items().deref_mut_trait()
                && infcx
                    .type_implements_trait(deref_mut_did, iter::once(found_ty), param_env)
                    .must_apply_modulo_regions()
            {
                Some(("call `Option::as_deref_mut()` first", ".as_deref_mut()"))
            } else if let hir::Mutability::Not = needs_mut {
                Some(("call `Option::as_deref()` first", ".as_deref()"))
            } else {
                None
            };

            if let Some((msg, sugg)) = help {
                err.span_suggestion_with_style(
                    expr.span.shrink_to_hi(),
                    msg,
                    sugg,
                    Applicability::MaybeIncorrect,
                    SuggestionStyle::ShowAlways,
                );
            }
        }
    }

    fn look_for_iterator_item_mistakes<G: EmissionGuarantee>(
        &self,
        assocs_in_this_method: &[Option<(Span, (DefId, Ty<'tcx>))>],
        typeck_results: &TypeckResults<'tcx>,
        type_diffs: &[TypeError<'tcx>],
        param_env: ty::ParamEnv<'tcx>,
        path_segment: &hir::PathSegment<'_>,
        args: &[hir::Expr<'_>],
        err: &mut Diag<'_, G>,
    ) {
        let tcx = self.tcx;
        // Special case for iterator chains, we look at potential failures of `Iterator::Item`
        // not being `: Clone` and `Iterator::map` calls with spurious trailing `;`.
        for entry in assocs_in_this_method {
            let Some((_span, (def_id, ty))) = entry else {
                continue;
            };
            for diff in type_diffs {
                let TypeError::Sorts(expected_found) = diff else {
                    continue;
                };
                if tcx.is_diagnostic_item(sym::IteratorItem, *def_id)
                    && path_segment.ident.name == sym::map
                    && self.can_eq(param_env, expected_found.found, *ty)
                    && let [arg] = args
                    && let hir::ExprKind::Closure(closure) = arg.kind
                {
                    let body = tcx.hir_body(closure.body);
                    if let hir::ExprKind::Block(block, None) = body.value.kind
                        && let None = block.expr
                        && let [.., stmt] = block.stmts
                        && let hir::StmtKind::Semi(expr) = stmt.kind
                        // FIXME: actually check the expected vs found types, but right now
                        // the expected is a projection that we need to resolve.
                        // && let Some(tail_ty) = typeck_results.expr_ty_opt(expr)
                        && expected_found.found.is_unit()
                        // FIXME: this happens with macro calls. Need to figure out why the stmt
                        // `println!();` doesn't include the `;` in its `Span`. (#133845)
                        // We filter these out to avoid ICEs with debug assertions on caused by
                        // empty suggestions.
                        && expr.span.hi() != stmt.span.hi()
                    {
                        err.span_suggestion_verbose(
                            expr.span.shrink_to_hi().with_hi(stmt.span.hi()),
                            "consider removing this semicolon",
                            String::new(),
                            Applicability::MachineApplicable,
                        );
                    }
                    let expr = if let hir::ExprKind::Block(block, None) = body.value.kind
                        && let Some(expr) = block.expr
                    {
                        expr
                    } else {
                        body.value
                    };
                    if let hir::ExprKind::MethodCall(path_segment, rcvr, [], span) = expr.kind
                        && path_segment.ident.name == sym::clone
                        && let Some(expr_ty) = typeck_results.expr_ty_opt(expr)
                        && let Some(rcvr_ty) = typeck_results.expr_ty_opt(rcvr)
                        && self.can_eq(param_env, expr_ty, rcvr_ty)
                        && let ty::Ref(_, ty, _) = expr_ty.kind()
                    {
                        err.span_label(
                            span,
                            format!(
                                "this method call is cloning the reference `{expr_ty}`, not \
                                 `{ty}` which doesn't implement `Clone`",
                            ),
                        );
                        let ty::Param(..) = ty.kind() else {
                            continue;
                        };
                        let node =
                            tcx.hir_node_by_def_id(tcx.hir_get_parent_item(expr.hir_id).def_id);

                        let pred = ty::Binder::dummy(ty::TraitPredicate {
                            trait_ref: ty::TraitRef::new(
                                tcx,
                                tcx.require_lang_item(LangItem::Clone, span),
                                [*ty],
                            ),
                            polarity: ty::PredicatePolarity::Positive,
                        });
                        let Some(generics) = node.generics() else {
                            continue;
                        };
                        let Some(body_id) = node.body_id() else {
                            continue;
                        };
                        suggest_restriction(
                            tcx,
                            tcx.hir_body_owner_def_id(body_id),
                            generics,
                            &format!("type parameter `{ty}`"),
                            err,
                            node.fn_sig(),
                            None,
                            pred,
                            None,
                        );
                    }
                }
            }
        }
    }

    fn point_at_chain<G: EmissionGuarantee>(
        &self,
        expr: &hir::Expr<'_>,
        typeck_results: &TypeckResults<'tcx>,
        type_diffs: Vec<TypeError<'tcx>>,
        param_env: ty::ParamEnv<'tcx>,
        err: &mut Diag<'_, G>,
    ) {
        let mut primary_spans = vec![];
        let mut span_labels = vec![];

        let tcx = self.tcx;

        let mut print_root_expr = true;
        let mut assocs = vec![];
        let mut expr = expr;
        let mut prev_ty = self.resolve_vars_if_possible(
            typeck_results.expr_ty_adjusted_opt(expr).unwrap_or(Ty::new_misc_error(tcx)),
        );
        while let hir::ExprKind::MethodCall(path_segment, rcvr_expr, args, span) = expr.kind {
            // Point at every method call in the chain with the resulting type.
            // vec![1, 2, 3].iter().map(mapper).sum<i32>()
            //               ^^^^^^ ^^^^^^^^^^^
            expr = rcvr_expr;
            let assocs_in_this_method =
                self.probe_assoc_types_at_expr(&type_diffs, span, prev_ty, expr.hir_id, param_env);
            self.look_for_iterator_item_mistakes(
                &assocs_in_this_method,
                typeck_results,
                &type_diffs,
                param_env,
                path_segment,
                args,
                err,
            );
            assocs.push(assocs_in_this_method);
            prev_ty = self.resolve_vars_if_possible(
                typeck_results.expr_ty_adjusted_opt(expr).unwrap_or(Ty::new_misc_error(tcx)),
            );

            if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = expr.kind
                && let hir::Path { res: Res::Local(hir_id), .. } = path
                && let hir::Node::Pat(binding) = self.tcx.hir_node(*hir_id)
            {
                let parent = self.tcx.parent_hir_node(binding.hir_id);
                // We've reached the root of the method call chain...
                if let hir::Node::LetStmt(local) = parent
                    && let Some(binding_expr) = local.init
                {
                    // ...and it is a binding. Get the binding creation and continue the chain.
                    expr = binding_expr;
                }
                if let hir::Node::Param(param) = parent {
                    // ...and it is an fn argument.
                    let prev_ty = self.resolve_vars_if_possible(
                        typeck_results
                            .node_type_opt(param.hir_id)
                            .unwrap_or(Ty::new_misc_error(tcx)),
                    );
                    let assocs_in_this_method = self.probe_assoc_types_at_expr(
                        &type_diffs,
                        param.ty_span,
                        prev_ty,
                        param.hir_id,
                        param_env,
                    );
                    if assocs_in_this_method.iter().any(|a| a.is_some()) {
                        assocs.push(assocs_in_this_method);
                        print_root_expr = false;
                    }
                    break;
                }
            }
        }
        // We want the type before deref coercions, otherwise we talk about `&[_]`
        // instead of `Vec<_>`.
        if let Some(ty) = typeck_results.expr_ty_opt(expr)
            && print_root_expr
        {
            let ty = with_forced_trimmed_paths!(self.ty_to_string(ty));
            // Point at the root expression
            // vec![1, 2, 3].iter().map(mapper).sum<i32>()
            // ^^^^^^^^^^^^^
            span_labels.push((expr.span, format!("this expression has type `{ty}`")));
        };
        // Only show this if it is not a "trivial" expression (not a method
        // chain) and there are associated types to talk about.
        let mut assocs = assocs.into_iter().peekable();
        while let Some(assocs_in_method) = assocs.next() {
            let Some(prev_assoc_in_method) = assocs.peek() else {
                for entry in assocs_in_method {
                    let Some((span, (assoc, ty))) = entry else {
                        continue;
                    };
                    if primary_spans.is_empty()
                        || type_diffs.iter().any(|diff| {
                            let TypeError::Sorts(expected_found) = diff else {
                                return false;
                            };
                            self.can_eq(param_env, expected_found.found, ty)
                        })
                    {
                        // FIXME: this doesn't quite work for `Iterator::collect`
                        // because we have `Vec<i32>` and `()`, but we'd want `i32`
                        // to point at the `.into_iter()` call, but as long as we
                        // still point at the other method calls that might have
                        // introduced the issue, this is fine for now.
                        primary_spans.push(span);
                    }
                    span_labels.push((
                        span,
                        with_forced_trimmed_paths!(format!(
                            "`{}` is `{ty}` here",
                            self.tcx.def_path_str(assoc),
                        )),
                    ));
                }
                break;
            };
            for (entry, prev_entry) in
                assocs_in_method.into_iter().zip(prev_assoc_in_method.into_iter())
            {
                match (entry, prev_entry) {
                    (Some((span, (assoc, ty))), Some((_, (_, prev_ty)))) => {
                        let ty_str = with_forced_trimmed_paths!(self.ty_to_string(ty));

                        let assoc = with_forced_trimmed_paths!(self.tcx.def_path_str(assoc));
                        if !self.can_eq(param_env, ty, *prev_ty) {
                            if type_diffs.iter().any(|diff| {
                                let TypeError::Sorts(expected_found) = diff else {
                                    return false;
                                };
                                self.can_eq(param_env, expected_found.found, ty)
                            }) {
                                primary_spans.push(span);
                            }
                            span_labels
                                .push((span, format!("`{assoc}` changed to `{ty_str}` here")));
                        } else {
                            span_labels.push((span, format!("`{assoc}` remains `{ty_str}` here")));
                        }
                    }
                    (Some((span, (assoc, ty))), None) => {
                        span_labels.push((
                            span,
                            with_forced_trimmed_paths!(format!(
                                "`{}` is `{}` here",
                                self.tcx.def_path_str(assoc),
                                self.ty_to_string(ty),
                            )),
                        ));
                    }
                    (None, Some(_)) | (None, None) => {}
                }
            }
        }
        if !primary_spans.is_empty() {
            let mut multi_span: MultiSpan = primary_spans.into();
            for (span, label) in span_labels {
                multi_span.push_span_label(span, label);
            }
            err.span_note(
                multi_span,
                "the method call chain might not have had the expected associated types",
            );
        }
    }

    fn probe_assoc_types_at_expr(
        &self,
        type_diffs: &[TypeError<'tcx>],
        span: Span,
        prev_ty: Ty<'tcx>,
        body_id: HirId,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Vec<Option<(Span, (DefId, Ty<'tcx>))>> {
        let ocx = ObligationCtxt::new(self.infcx);
        let mut assocs_in_this_method = Vec::with_capacity(type_diffs.len());
        for diff in type_diffs {
            let TypeError::Sorts(expected_found) = diff else {
                continue;
            };
            let ty::Alias(ty::Projection, proj) = expected_found.expected.kind() else {
                continue;
            };

            // Make `Self` be equivalent to the type of the call chain
            // expression we're looking at now, so that we can tell what
            // for example `Iterator::Item` is at this point in the chain.
            let args = GenericArgs::for_item(self.tcx, proj.def_id, |param, _| {
                if param.index == 0 {
                    debug_assert_matches!(param.kind, ty::GenericParamDefKind::Type { .. });
                    return prev_ty.into();
                }
                self.var_for_def(span, param)
            });
            // This will hold the resolved type of the associated type, if the
            // current expression implements the trait that associated type is
            // in. For example, this would be what `Iterator::Item` is here.
            let ty = self.infcx.next_ty_var(span);
            // This corresponds to `<ExprTy as Iterator>::Item = _`.
            let projection = ty::Binder::dummy(ty::PredicateKind::Clause(
                ty::ClauseKind::Projection(ty::ProjectionPredicate {
                    projection_term: ty::AliasTerm::new_from_args(self.tcx, proj.def_id, args),
                    term: ty.into(),
                }),
            ));
            let body_def_id = self.tcx.hir_enclosing_body_owner(body_id);
            // Add `<ExprTy as Iterator>::Item = _` obligation.
            ocx.register_obligation(Obligation::misc(
                self.tcx,
                span,
                body_def_id,
                param_env,
                projection,
            ));
            if ocx.select_where_possible().is_empty()
                && let ty = self.resolve_vars_if_possible(ty)
                && !ty.is_ty_var()
            {
                assocs_in_this_method.push(Some((span, (proj.def_id, ty))));
            } else {
                // `<ExprTy as Iterator>` didn't select, so likely we've
                // reached the end of the iterator chain, like the originating
                // `Vec<_>` or the `ty` couldn't be determined.
                // Keep the space consistent for later zipping.
                assocs_in_this_method.push(None);
            }
        }
        assocs_in_this_method
    }

    /// If the type that failed selection is an array or a reference to an array,
    /// but the trait is implemented for slices, suggest that the user converts
    /// the array into a slice.
    pub(super) fn suggest_convert_to_slice(
        &self,
        err: &mut Diag<'_>,
        obligation: &PredicateObligation<'tcx>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        candidate_impls: &[ImplCandidate<'tcx>],
        span: Span,
    ) {
        // We can only suggest the slice coersion for function and binary operation arguments,
        // since the suggestion would make no sense in turbofish or call
        let (ObligationCauseCode::BinOp { .. } | ObligationCauseCode::FunctionArg { .. }) =
            obligation.cause.code()
        else {
            return;
        };

        // Three cases where we can make a suggestion:
        // 1. `[T; _]` (array of T)
        // 2. `&[T; _]` (reference to array of T)
        // 3. `&mut [T; _]` (mutable reference to array of T)
        let (element_ty, mut mutability) = match *trait_pred.skip_binder().self_ty().kind() {
            ty::Array(element_ty, _) => (element_ty, None),

            ty::Ref(_, pointee_ty, mutability) => match *pointee_ty.kind() {
                ty::Array(element_ty, _) => (element_ty, Some(mutability)),
                _ => return,
            },

            _ => return,
        };

        // Go through all the candidate impls to see if any of them is for
        // slices of `element_ty` with `mutability`.
        let mut is_slice = |candidate: Ty<'tcx>| match *candidate.kind() {
            ty::RawPtr(t, m) | ty::Ref(_, t, m) => {
                if matches!(*t.kind(), ty::Slice(e) if e == element_ty)
                    && m == mutability.unwrap_or(m)
                {
                    // Use the candidate's mutability going forward.
                    mutability = Some(m);
                    true
                } else {
                    false
                }
            }
            _ => false,
        };

        // Grab the first candidate that matches, if any, and make a suggestion.
        if let Some(slice_ty) = candidate_impls
            .iter()
            .map(|trait_ref| trait_ref.trait_ref.self_ty())
            .find(|t| is_slice(*t))
        {
            let msg = format!("convert the array to a `{slice_ty}` slice instead");

            if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                let mut suggestions = vec![];
                if snippet.starts_with('&') {
                } else if let Some(hir::Mutability::Mut) = mutability {
                    suggestions.push((span.shrink_to_lo(), "&mut ".into()));
                } else {
                    suggestions.push((span.shrink_to_lo(), "&".into()));
                }
                suggestions.push((span.shrink_to_hi(), "[..]".into()));
                err.multipart_suggestion_verbose(msg, suggestions, Applicability::MaybeIncorrect);
            } else {
                err.span_help(span, msg);
            }
        }
    }

    /// If the type failed selection but the trait is implemented for `(T,)`, suggest that the user
    /// creates a unary tuple
    ///
    /// This is a common gotcha when using libraries that emulate variadic functions with traits for tuples.
    pub(super) fn suggest_tuple_wrapping(
        &self,
        err: &mut Diag<'_>,
        root_obligation: &PredicateObligation<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) {
        let ObligationCauseCode::FunctionArg { arg_hir_id, .. } = obligation.cause.code() else {
            return;
        };

        let Some(root_pred) = root_obligation.predicate.as_trait_clause() else { return };

        let trait_ref = root_pred.map_bound(|root_pred| {
            root_pred
                .trait_ref
                .with_self_ty(self.tcx, Ty::new_tup(self.tcx, &[root_pred.trait_ref.self_ty()]))
        });

        let obligation =
            Obligation::new(self.tcx, obligation.cause.clone(), obligation.param_env, trait_ref);

        if self.predicate_must_hold_modulo_regions(&obligation) {
            let arg_span = self.tcx.hir_span(*arg_hir_id);
            err.multipart_suggestion_verbose(
                format!("use a unary tuple instead"),
                vec![(arg_span.shrink_to_lo(), "(".into()), (arg_span.shrink_to_hi(), ",)".into())],
                Applicability::MaybeIncorrect,
            );
        }
    }

    pub(super) fn explain_hrtb_projection(
        &self,
        diag: &mut Diag<'_>,
        pred: ty::PolyTraitPredicate<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        cause: &ObligationCause<'tcx>,
    ) {
        if pred.skip_binder().has_escaping_bound_vars() && pred.skip_binder().has_non_region_infer()
        {
            self.probe(|_| {
                let ocx = ObligationCtxt::new(self);
                self.enter_forall(pred, |pred| {
                    let pred = ocx.normalize(&ObligationCause::dummy(), param_env, pred);
                    ocx.register_obligation(Obligation::new(
                        self.tcx,
                        ObligationCause::dummy(),
                        param_env,
                        pred,
                    ));
                });
                if !ocx.select_where_possible().is_empty() {
                    // encountered errors.
                    return;
                }

                if let ObligationCauseCode::FunctionArg {
                    call_hir_id,
                    arg_hir_id,
                    parent_code: _,
                } = cause.code()
                {
                    let arg_span = self.tcx.hir_span(*arg_hir_id);
                    let mut sp: MultiSpan = arg_span.into();

                    sp.push_span_label(
                        arg_span,
                        "the trait solver is unable to infer the \
                        generic types that should be inferred from this argument",
                    );
                    sp.push_span_label(
                        self.tcx.hir_span(*call_hir_id),
                        "add turbofish arguments to this call to \
                        specify the types manually, even if it's redundant",
                    );
                    diag.span_note(
                        sp,
                        "this is a known limitation of the trait solver that \
                        will be lifted in the future",
                    );
                } else {
                    let mut sp: MultiSpan = cause.span.into();
                    sp.push_span_label(
                        cause.span,
                        "try adding turbofish arguments to this expression to \
                        specify the types manually, even if it's redundant",
                    );
                    diag.span_note(
                        sp,
                        "this is a known limitation of the trait solver that \
                        will be lifted in the future",
                    );
                }
            });
        }
    }

    pub(super) fn suggest_desugaring_async_fn_in_trait(
        &self,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) {
        // Don't suggest if RTN is active -- we should prefer a where-clause bound instead.
        if self.tcx.features().return_type_notation() {
            return;
        }

        let trait_def_id = trait_pred.def_id();

        // Only suggest specifying auto traits
        if !self.tcx.trait_is_auto(trait_def_id) {
            return;
        }

        // Look for an RPITIT
        let ty::Alias(ty::Projection, alias_ty) = trait_pred.self_ty().skip_binder().kind() else {
            return;
        };
        let Some(ty::ImplTraitInTraitData::Trait { fn_def_id, opaque_def_id }) =
            self.tcx.opt_rpitit_info(alias_ty.def_id)
        else {
            return;
        };

        let auto_trait = self.tcx.def_path_str(trait_def_id);
        // ... which is a local function
        let Some(fn_def_id) = fn_def_id.as_local() else {
            // If it's not local, we can at least mention that the method is async, if it is.
            if self.tcx.asyncness(fn_def_id).is_async() {
                err.span_note(
                    self.tcx.def_span(fn_def_id),
                    format!(
                        "`{}::{}` is an `async fn` in trait, which does not \
                    automatically imply that its future is `{auto_trait}`",
                        alias_ty.trait_ref(self.tcx),
                        self.tcx.item_name(fn_def_id)
                    ),
                );
            }
            return;
        };
        let hir::Node::TraitItem(item) = self.tcx.hir_node_by_def_id(fn_def_id) else {
            return;
        };

        // ... whose signature is `async` (i.e. this is an AFIT)
        let (sig, body) = item.expect_fn();
        let hir::FnRetTy::Return(hir::Ty { kind: hir::TyKind::OpaqueDef(opaq_def, ..), .. }) =
            sig.decl.output
        else {
            // This should never happen, but let's not ICE.
            return;
        };

        // Check that this is *not* a nested `impl Future` RPIT in an async fn
        // (i.e. `async fn foo() -> impl Future`)
        if opaq_def.def_id.to_def_id() != opaque_def_id {
            return;
        }

        let Some(sugg) = suggest_desugaring_async_fn_to_impl_future_in_trait(
            self.tcx,
            *sig,
            *body,
            opaque_def_id.expect_local(),
            &format!(" + {auto_trait}"),
        ) else {
            return;
        };

        let function_name = self.tcx.def_path_str(fn_def_id);
        err.multipart_suggestion(
            format!(
                "`{auto_trait}` can be made part of the associated future's \
                guarantees for all implementations of `{function_name}`"
            ),
            sugg,
            Applicability::MachineApplicable,
        );
    }

    pub fn ty_kind_suggestion(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<String> {
        let tcx = self.infcx.tcx;
        let implements_default = |ty| {
            let Some(default_trait) = tcx.get_diagnostic_item(sym::Default) else {
                return false;
            };
            self.type_implements_trait(default_trait, [ty], param_env).must_apply_modulo_regions()
        };

        Some(match *ty.kind() {
            ty::Never | ty::Error(_) => return None,
            ty::Bool => "false".to_string(),
            ty::Char => "\'x\'".to_string(),
            ty::Int(_) | ty::Uint(_) => "42".into(),
            ty::Float(_) => "3.14159".into(),
            ty::Slice(_) => "[]".to_string(),
            ty::Adt(def, _) if Some(def.did()) == tcx.get_diagnostic_item(sym::Vec) => {
                "vec![]".to_string()
            }
            ty::Adt(def, _) if Some(def.did()) == tcx.get_diagnostic_item(sym::String) => {
                "String::new()".to_string()
            }
            ty::Adt(def, args) if def.is_box() => {
                format!("Box::new({})", self.ty_kind_suggestion(param_env, args[0].expect_ty())?)
            }
            ty::Adt(def, _) if Some(def.did()) == tcx.get_diagnostic_item(sym::Option) => {
                "None".to_string()
            }
            ty::Adt(def, args) if Some(def.did()) == tcx.get_diagnostic_item(sym::Result) => {
                format!("Ok({})", self.ty_kind_suggestion(param_env, args[0].expect_ty())?)
            }
            ty::Adt(_, _) if implements_default(ty) => "Default::default()".to_string(),
            ty::Ref(_, ty, mutability) => {
                if let (ty::Str, hir::Mutability::Not) = (ty.kind(), mutability) {
                    "\"\"".to_string()
                } else {
                    let ty = self.ty_kind_suggestion(param_env, ty)?;
                    format!("&{}{ty}", mutability.prefix_str())
                }
            }
            ty::Array(ty, len) if let Some(len) = len.try_to_target_usize(tcx) => {
                if len == 0 {
                    "[]".to_string()
                } else if self.type_is_copy_modulo_regions(param_env, ty) || len == 1 {
                    // Can only suggest `[ty; 0]` if sz == 1 or copy
                    format!("[{}; {}]", self.ty_kind_suggestion(param_env, ty)?, len)
                } else {
                    "/* value */".to_string()
                }
            }
            ty::Tuple(tys) => format!(
                "({}{})",
                tys.iter()
                    .map(|ty| self.ty_kind_suggestion(param_env, ty))
                    .collect::<Option<Vec<String>>>()?
                    .join(", "),
                if tys.len() == 1 { "," } else { "" }
            ),
            _ => "/* value */".to_string(),
        })
    }

    // For E0277 when use `?` operator, suggest adding
    // a suitable return type in `FnSig`, and a default
    // return value at the end of the function's body.
    pub(super) fn suggest_add_result_as_return_type(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diag<'_>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) {
        if ObligationCauseCode::QuestionMark != *obligation.cause.code().peel_derives() {
            return;
        }

        // Only suggest for local function and associated method,
        // because this suggest adding both return type in
        // the `FnSig` and a default return value in the body, so it
        // is not suitable for foreign function without a local body,
        // and neither for trait method which may be also implemented
        // in other place, so shouldn't change it's FnSig.
        fn choose_suggest_items<'tcx, 'hir>(
            tcx: TyCtxt<'tcx>,
            node: hir::Node<'hir>,
        ) -> Option<(&'hir hir::FnDecl<'hir>, hir::BodyId)> {
            match node {
                hir::Node::Item(item)
                    if let hir::ItemKind::Fn { sig, body: body_id, .. } = item.kind =>
                {
                    Some((sig.decl, body_id))
                }
                hir::Node::ImplItem(item)
                    if let hir::ImplItemKind::Fn(sig, body_id) = item.kind =>
                {
                    let parent = tcx.parent_hir_node(item.hir_id());
                    if let hir::Node::Item(item) = parent
                        && let hir::ItemKind::Impl(imp) = item.kind
                        && imp.of_trait.is_none()
                    {
                        return Some((sig.decl, body_id));
                    }
                    None
                }
                _ => None,
            }
        }

        let node = self.tcx.hir_node_by_def_id(obligation.cause.body_id);
        if let Some((fn_decl, body_id)) = choose_suggest_items(self.tcx, node)
            && let hir::FnRetTy::DefaultReturn(ret_span) = fn_decl.output
            && self.tcx.is_diagnostic_item(sym::FromResidual, trait_pred.def_id())
            && trait_pred.skip_binder().trait_ref.args.type_at(0).is_unit()
            && let ty::Adt(def, _) = trait_pred.skip_binder().trait_ref.args.type_at(1).kind()
            && self.tcx.is_diagnostic_item(sym::Result, def.did())
        {
            let mut sugg_spans =
                vec![(ret_span, " -> Result<(), Box<dyn std::error::Error>>".to_string())];
            let body = self.tcx.hir_body(body_id);
            if let hir::ExprKind::Block(b, _) = body.value.kind
                && b.expr.is_none()
            {
                // The span of '}' in the end of block.
                let span = self.tcx.sess.source_map().end_point(b.span);
                sugg_spans.push((
                    span.shrink_to_lo(),
                    format!(
                        "{}{}",
                        "    Ok(())\n",
                        self.tcx.sess.source_map().indentation_before(span).unwrap_or_default(),
                    ),
                ));
            }
            err.multipart_suggestion_verbose(
                format!("consider adding return type"),
                sugg_spans,
                Applicability::MaybeIncorrect,
            );
        }
    }

    #[instrument(level = "debug", skip_all)]
    pub(super) fn suggest_unsized_bound_if_applicable(
        &self,
        err: &mut Diag<'_>,
        obligation: &PredicateObligation<'tcx>,
    ) {
        let ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) =
            obligation.predicate.kind().skip_binder()
        else {
            return;
        };
        let (ObligationCauseCode::WhereClause(item_def_id, span)
        | ObligationCauseCode::WhereClauseInExpr(item_def_id, span, ..)) =
            *obligation.cause.code().peel_derives()
        else {
            return;
        };
        if span.is_dummy() {
            return;
        }
        debug!(?pred, ?item_def_id, ?span);

        let (Some(node), true) = (
            self.tcx.hir_get_if_local(item_def_id),
            self.tcx.is_lang_item(pred.def_id(), LangItem::Sized),
        ) else {
            return;
        };

        let Some(generics) = node.generics() else {
            return;
        };
        let sized_trait = self.tcx.lang_items().sized_trait();
        debug!(?generics.params);
        debug!(?generics.predicates);
        let Some(param) = generics.params.iter().find(|param| param.span == span) else {
            return;
        };
        // Check that none of the explicit trait bounds is `Sized`. Assume that an explicit
        // `Sized` bound is there intentionally and we don't need to suggest relaxing it.
        let explicitly_sized = generics
            .bounds_for_param(param.def_id)
            .flat_map(|bp| bp.bounds)
            .any(|bound| bound.trait_ref().and_then(|tr| tr.trait_def_id()) == sized_trait);
        if explicitly_sized {
            return;
        }
        debug!(?param);
        match node {
            hir::Node::Item(
                item @ hir::Item {
                    // Only suggest indirection for uses of type parameters in ADTs.
                    kind:
                        hir::ItemKind::Enum(..) | hir::ItemKind::Struct(..) | hir::ItemKind::Union(..),
                    ..
                },
            ) => {
                if self.suggest_indirection_for_unsized(err, item, param) {
                    return;
                }
            }
            _ => {}
        };

        // Didn't add an indirection suggestion, so add a general suggestion to relax `Sized`.
        let (span, separator, open_paren_sp) =
            if let Some((s, open_paren_sp)) = generics.bounds_span_for_suggestions(param.def_id) {
                (s, " +", open_paren_sp)
            } else {
                (param.name.ident().span.shrink_to_hi(), ":", None)
            };

        let mut suggs = vec![];
        let suggestion = format!("{separator} ?Sized");

        if let Some(open_paren_sp) = open_paren_sp {
            suggs.push((open_paren_sp, "(".to_string()));
            suggs.push((span, format!("){suggestion}")));
        } else {
            suggs.push((span, suggestion));
        }

        err.multipart_suggestion_verbose(
            "consider relaxing the implicit `Sized` restriction",
            suggs,
            Applicability::MachineApplicable,
        );
    }

    fn suggest_indirection_for_unsized(
        &self,
        err: &mut Diag<'_>,
        item: &hir::Item<'tcx>,
        param: &hir::GenericParam<'tcx>,
    ) -> bool {
        // Suggesting `T: ?Sized` is only valid in an ADT if `T` is only used in a
        // borrow. `struct S<'a, T: ?Sized>(&'a T);` is valid, `struct S<T: ?Sized>(T);`
        // is not. Look for invalid "bare" parameter uses, and suggest using indirection.
        let mut visitor =
            FindTypeParam { param: param.name.ident().name, invalid_spans: vec![], nested: false };
        visitor.visit_item(item);
        if visitor.invalid_spans.is_empty() {
            return false;
        }
        let mut multispan: MultiSpan = param.span.into();
        multispan.push_span_label(
            param.span,
            format!("this could be changed to `{}: ?Sized`...", param.name.ident()),
        );
        for sp in visitor.invalid_spans {
            multispan.push_span_label(
                sp,
                format!("...if indirection were used here: `Box<{}>`", param.name.ident()),
            );
        }
        err.span_help(
            multispan,
            format!(
                "you could relax the implicit `Sized` bound on `{T}` if it were \
                used through indirection like `&{T}` or `Box<{T}>`",
                T = param.name.ident(),
            ),
        );
        true
    }
    pub(crate) fn suggest_swapping_lhs_and_rhs<T>(
        &self,
        err: &mut Diag<'_>,
        predicate: T,
        param_env: ty::ParamEnv<'tcx>,
        cause_code: &ObligationCauseCode<'tcx>,
    ) where
        T: Upcast<TyCtxt<'tcx>, ty::Predicate<'tcx>>,
    {
        let tcx = self.tcx;
        let predicate = predicate.upcast(tcx);
        match *cause_code {
            ObligationCauseCode::BinOp {
                lhs_hir_id,
                rhs_hir_id: Some(rhs_hir_id),
                rhs_span: Some(rhs_span),
                ..
            } if let Some(typeck_results) = &self.typeck_results
                && let hir::Node::Expr(lhs) = tcx.hir_node(lhs_hir_id)
                && let hir::Node::Expr(rhs) = tcx.hir_node(rhs_hir_id)
                && let Some(lhs_ty) = typeck_results.expr_ty_opt(lhs)
                && let Some(rhs_ty) = typeck_results.expr_ty_opt(rhs) =>
            {
                if let Some(pred) = predicate.as_trait_clause()
                    && tcx.is_lang_item(pred.def_id(), LangItem::PartialEq)
                    && self
                        .infcx
                        .type_implements_trait(pred.def_id(), [rhs_ty, lhs_ty], param_env)
                        .must_apply_modulo_regions()
                {
                    let lhs_span = tcx.hir_span(lhs_hir_id);
                    let sm = tcx.sess.source_map();
                    if let Ok(rhs_snippet) = sm.span_to_snippet(rhs_span)
                        && let Ok(lhs_snippet) = sm.span_to_snippet(lhs_span)
                    {
                        err.note(format!("`{rhs_ty}` implements `PartialEq<{lhs_ty}>`"));
                        err.multipart_suggestion(
                            "consider swapping the equality",
                            vec![(lhs_span, rhs_snippet), (rhs_span, lhs_snippet)],
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

/// Add a hint to add a missing borrow or remove an unnecessary one.
fn hint_missing_borrow<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
    found: Ty<'tcx>,
    expected: Ty<'tcx>,
    found_node: Node<'_>,
    err: &mut Diag<'_>,
) {
    if matches!(found_node, Node::TraitItem(..)) {
        return;
    }

    let found_args = match found.kind() {
        ty::FnPtr(sig_tys, _) => infcx.enter_forall(*sig_tys, |sig_tys| sig_tys.inputs().iter()),
        kind => {
            span_bug!(span, "found was converted to a FnPtr above but is now {:?}", kind)
        }
    };
    let expected_args = match expected.kind() {
        ty::FnPtr(sig_tys, _) => infcx.enter_forall(*sig_tys, |sig_tys| sig_tys.inputs().iter()),
        kind => {
            span_bug!(span, "expected was converted to a FnPtr above but is now {:?}", kind)
        }
    };

    // This could be a variant constructor, for example.
    let Some(fn_decl) = found_node.fn_decl() else {
        return;
    };

    let args = fn_decl.inputs.iter();

    let mut to_borrow = Vec::new();
    let mut remove_borrow = Vec::new();

    for ((found_arg, expected_arg), arg) in found_args.zip(expected_args).zip(args) {
        let (found_ty, found_refs) = get_deref_type_and_refs(*found_arg);
        let (expected_ty, expected_refs) = get_deref_type_and_refs(*expected_arg);

        if infcx.can_eq(param_env, found_ty, expected_ty) {
            // FIXME: This could handle more exotic cases like mutability mismatches too!
            if found_refs.len() < expected_refs.len()
                && found_refs[..] == expected_refs[expected_refs.len() - found_refs.len()..]
            {
                to_borrow.push((
                    arg.span.shrink_to_lo(),
                    expected_refs[..expected_refs.len() - found_refs.len()]
                        .iter()
                        .map(|mutbl| format!("&{}", mutbl.prefix_str()))
                        .collect::<Vec<_>>()
                        .join(""),
                ));
            } else if found_refs.len() > expected_refs.len() {
                let mut span = arg.span.shrink_to_lo();
                let mut left = found_refs.len() - expected_refs.len();
                let mut ty = arg;
                while let hir::TyKind::Ref(_, mut_ty) = &ty.kind
                    && left > 0
                {
                    span = span.with_hi(mut_ty.ty.span.lo());
                    ty = mut_ty.ty;
                    left -= 1;
                }
                let sugg = if left == 0 {
                    (span, String::new())
                } else {
                    (arg.span, expected_arg.to_string())
                };
                remove_borrow.push(sugg);
            }
        }
    }

    if !to_borrow.is_empty() {
        err.subdiagnostic(errors::AdjustSignatureBorrow::Borrow { to_borrow });
    }

    if !remove_borrow.is_empty() {
        err.subdiagnostic(errors::AdjustSignatureBorrow::RemoveBorrow { remove_borrow });
    }
}

/// Collect all the paths that reference `Self`.
/// Used to suggest replacing associated types with an explicit type in `where` clauses.
#[derive(Debug)]
pub struct SelfVisitor<'v> {
    pub paths: Vec<&'v hir::Ty<'v>>,
    pub name: Option<Symbol>,
}

impl<'v> Visitor<'v> for SelfVisitor<'v> {
    fn visit_ty(&mut self, ty: &'v hir::Ty<'v, AmbigArg>) {
        if let hir::TyKind::Path(path) = ty.kind
            && let hir::QPath::TypeRelative(inner_ty, segment) = path
            && (Some(segment.ident.name) == self.name || self.name.is_none())
            && let hir::TyKind::Path(inner_path) = inner_ty.kind
            && let hir::QPath::Resolved(None, inner_path) = inner_path
            && let Res::SelfTyAlias { .. } = inner_path.res
        {
            self.paths.push(ty.as_unambig_ty());
        }
        hir::intravisit::walk_ty(self, ty);
    }
}

/// Collect all the returned expressions within the input expression.
/// Used to point at the return spans when we want to suggest some change to them.
#[derive(Default)]
pub struct ReturnsVisitor<'v> {
    pub returns: Vec<&'v hir::Expr<'v>>,
    in_block_tail: bool,
}

impl<'v> Visitor<'v> for ReturnsVisitor<'v> {
    fn visit_expr(&mut self, ex: &'v hir::Expr<'v>) {
        // Visit every expression to detect `return` paths, either through the function's tail
        // expression or `return` statements. We walk all nodes to find `return` statements, but
        // we only care about tail expressions when `in_block_tail` is `true`, which means that
        // they're in the return path of the function body.
        match ex.kind {
            hir::ExprKind::Ret(Some(ex)) => {
                self.returns.push(ex);
            }
            hir::ExprKind::Block(block, _) if self.in_block_tail => {
                self.in_block_tail = false;
                for stmt in block.stmts {
                    hir::intravisit::walk_stmt(self, stmt);
                }
                self.in_block_tail = true;
                if let Some(expr) = block.expr {
                    self.visit_expr(expr);
                }
            }
            hir::ExprKind::If(_, then, else_opt) if self.in_block_tail => {
                self.visit_expr(then);
                if let Some(el) = else_opt {
                    self.visit_expr(el);
                }
            }
            hir::ExprKind::Match(_, arms, _) if self.in_block_tail => {
                for arm in arms {
                    self.visit_expr(arm.body);
                }
            }
            // We need to walk to find `return`s in the entire body.
            _ if !self.in_block_tail => hir::intravisit::walk_expr(self, ex),
            _ => self.returns.push(ex),
        }
    }

    fn visit_body(&mut self, body: &hir::Body<'v>) {
        assert!(!self.in_block_tail);
        self.in_block_tail = true;
        hir::intravisit::walk_body(self, body);
    }
}

/// Collect all the awaited expressions within the input expression.
#[derive(Default)]
struct AwaitsVisitor {
    awaits: Vec<HirId>,
}

impl<'v> Visitor<'v> for AwaitsVisitor {
    fn visit_expr(&mut self, ex: &'v hir::Expr<'v>) {
        if let hir::ExprKind::Yield(_, hir::YieldSource::Await { expr: Some(id) }) = ex.kind {
            self.awaits.push(id)
        }
        hir::intravisit::walk_expr(self, ex)
    }
}

/// Suggest a new type parameter name for diagnostic purposes.
///
/// `name` is the preferred name you'd like to suggest if it's not in use already.
pub trait NextTypeParamName {
    fn next_type_param_name(&self, name: Option<&str>) -> String;
}

impl NextTypeParamName for &[hir::GenericParam<'_>] {
    fn next_type_param_name(&self, name: Option<&str>) -> String {
        // Type names are usually single letters in uppercase. So convert the first letter of input string to uppercase.
        let name = name.and_then(|n| n.chars().next()).map(|c| c.to_uppercase().to_string());
        let name = name.as_deref();

        // This is the list of possible parameter names that we might suggest.
        let possible_names = [name.unwrap_or("T"), "T", "U", "V", "X", "Y", "Z", "A", "B", "C"];

        // Filter out used names based on `filter_fn`.
        let used_names: Vec<Symbol> = self
            .iter()
            .filter_map(|param| match param.name {
                hir::ParamName::Plain(ident) => Some(ident.name),
                _ => None,
            })
            .collect();

        // Find a name from `possible_names` that is not in `used_names`.
        possible_names
            .iter()
            .find(|n| !used_names.contains(&Symbol::intern(n)))
            .unwrap_or(&"ParamName")
            .to_string()
    }
}

/// Collect the spans that we see the generic param `param_did`
struct ReplaceImplTraitVisitor<'a> {
    ty_spans: &'a mut Vec<Span>,
    param_did: DefId,
}

impl<'a, 'hir> hir::intravisit::Visitor<'hir> for ReplaceImplTraitVisitor<'a> {
    fn visit_ty(&mut self, t: &'hir hir::Ty<'hir, AmbigArg>) {
        if let hir::TyKind::Path(hir::QPath::Resolved(
            None,
            hir::Path { res: Res::Def(_, segment_did), .. },
        )) = t.kind
        {
            if self.param_did == *segment_did {
                // `fn foo(t: impl Trait)`
                //            ^^^^^^^^^^ get this to suggest `T` instead

                // There might be more than one `impl Trait`.
                self.ty_spans.push(t.span);
                return;
            }
        }

        hir::intravisit::walk_ty(self, t);
    }
}

pub(super) fn get_explanation_based_on_obligation<'tcx>(
    tcx: TyCtxt<'tcx>,
    obligation: &PredicateObligation<'tcx>,
    trait_predicate: ty::PolyTraitPredicate<'tcx>,
    pre_message: String,
) -> String {
    if let ObligationCauseCode::MainFunctionType = obligation.cause.code() {
        "consider using `()`, or a `Result`".to_owned()
    } else {
        let ty_desc = match trait_predicate.self_ty().skip_binder().kind() {
            ty::FnDef(_, _) => Some("fn item"),
            ty::Closure(_, _) => Some("closure"),
            _ => None,
        };

        let desc = match ty_desc {
            Some(desc) => format!(" {desc}"),
            None => String::new(),
        };
        if let ty::PredicatePolarity::Positive = trait_predicate.polarity() {
            format!(
                "{pre_message}the trait `{}` is not implemented for{desc} `{}`",
                trait_predicate.print_modifiers_and_trait_path(),
                tcx.short_string(trait_predicate.self_ty().skip_binder(), &mut None),
            )
        } else {
            // "the trait bound `T: !Send` is not satisfied" reads better than "`!Send` is
            // not implemented for `T`".
            // FIXME: add note explaining explicit negative trait bounds.
            format!("{pre_message}the trait bound `{trait_predicate}` is not satisfied")
        }
    }
}

// Replace `param` with `replace_ty`
struct ReplaceImplTraitFolder<'tcx> {
    tcx: TyCtxt<'tcx>,
    param: &'tcx ty::GenericParamDef,
    replace_ty: Ty<'tcx>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ReplaceImplTraitFolder<'tcx> {
    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Param(ty::ParamTy { index, .. }) = t.kind() {
            if self.param.index == *index {
                return self.replace_ty;
            }
        }
        t.super_fold_with(self)
    }

    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

pub fn suggest_desugaring_async_fn_to_impl_future_in_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    sig: hir::FnSig<'tcx>,
    body: hir::TraitFn<'tcx>,
    opaque_def_id: LocalDefId,
    add_bounds: &str,
) -> Option<Vec<(Span, String)>> {
    let hir::IsAsync::Async(async_span) = sig.header.asyncness else {
        return None;
    };
    let async_span = tcx.sess.source_map().span_extend_while_whitespace(async_span);

    let future = tcx.hir_node_by_def_id(opaque_def_id).expect_opaque_ty();
    let [hir::GenericBound::Trait(trait_ref)] = future.bounds else {
        // `async fn` should always lower to a single bound... but don't ICE.
        return None;
    };
    let Some(hir::PathSegment { args: Some(args), .. }) = trait_ref.trait_ref.path.segments.last()
    else {
        // desugaring to a single path segment for `Future<...>`.
        return None;
    };
    let Some(future_output_ty) = args.constraints.first().and_then(|constraint| constraint.ty())
    else {
        // Also should never happen.
        return None;
    };

    let mut sugg = if future_output_ty.span.is_empty() {
        vec![
            (async_span, String::new()),
            (
                future_output_ty.span,
                format!(" -> impl std::future::Future<Output = ()>{add_bounds}"),
            ),
        ]
    } else {
        vec![
            (future_output_ty.span.shrink_to_lo(), "impl std::future::Future<Output = ".to_owned()),
            (future_output_ty.span.shrink_to_hi(), format!(">{add_bounds}")),
            (async_span, String::new()),
        ]
    };

    // If there's a body, we also need to wrap it in `async {}`
    if let hir::TraitFn::Provided(body) = body {
        let body = tcx.hir_body(body);
        let body_span = body.value.span;
        let body_span_without_braces =
            body_span.with_lo(body_span.lo() + BytePos(1)).with_hi(body_span.hi() - BytePos(1));
        if body_span_without_braces.is_empty() {
            sugg.push((body_span_without_braces, " async {} ".to_owned()));
        } else {
            sugg.extend([
                (body_span_without_braces.shrink_to_lo(), "async {".to_owned()),
                (body_span_without_braces.shrink_to_hi(), "} ".to_owned()),
            ]);
        }
    }

    Some(sugg)
}

/// On `impl` evaluation cycles, look for `Self::AssocTy` restrictions in `where` clauses, explain
/// they are not allowed and if possible suggest alternatives.
fn point_at_assoc_type_restriction<G: EmissionGuarantee>(
    tcx: TyCtxt<'_>,
    err: &mut Diag<'_, G>,
    self_ty_str: &str,
    trait_name: &str,
    predicate: ty::Predicate<'_>,
    generics: &hir::Generics<'_>,
    data: &ImplDerivedCause<'_>,
) {
    let ty::PredicateKind::Clause(clause) = predicate.kind().skip_binder() else {
        return;
    };
    let ty::ClauseKind::Projection(proj) = clause else {
        return;
    };
    let name = tcx.item_name(proj.projection_term.def_id);
    let mut predicates = generics.predicates.iter().peekable();
    let mut prev: Option<(&hir::WhereBoundPredicate<'_>, Span)> = None;
    while let Some(pred) = predicates.next() {
        let curr_span = pred.span;
        let hir::WherePredicateKind::BoundPredicate(pred) = pred.kind else {
            continue;
        };
        let mut bounds = pred.bounds.iter();
        while let Some(bound) = bounds.next() {
            let Some(trait_ref) = bound.trait_ref() else {
                continue;
            };
            if bound.span() != data.span {
                continue;
            }
            if let hir::TyKind::Path(path) = pred.bounded_ty.kind
                && let hir::QPath::TypeRelative(ty, segment) = path
                && segment.ident.name == name
                && let hir::TyKind::Path(inner_path) = ty.kind
                && let hir::QPath::Resolved(None, inner_path) = inner_path
                && let Res::SelfTyAlias { .. } = inner_path.res
            {
                // The following block is to determine the right span to delete for this bound
                // that will leave valid code after the suggestion is applied.
                let span = if pred.origin == hir::PredicateOrigin::WhereClause
                    && generics
                        .predicates
                        .iter()
                        .filter(|p| {
                            matches!(
                                p.kind,
                                hir::WherePredicateKind::BoundPredicate(p)
                                if hir::PredicateOrigin::WhereClause == p.origin
                            )
                        })
                        .count()
                        == 1
                {
                    // There's only one `where` bound, that needs to be removed. Remove the whole
                    // `where` clause.
                    generics.where_clause_span
                } else if let Some(next_pred) = predicates.peek()
                    && let hir::WherePredicateKind::BoundPredicate(next) = next_pred.kind
                    && pred.origin == next.origin
                {
                    // There's another bound, include the comma for the current one.
                    curr_span.until(next_pred.span)
                } else if let Some((prev, prev_span)) = prev
                    && pred.origin == prev.origin
                {
                    // Last bound, try to remove the previous comma.
                    prev_span.shrink_to_hi().to(curr_span)
                } else if pred.origin == hir::PredicateOrigin::WhereClause {
                    curr_span.with_hi(generics.where_clause_span.hi())
                } else {
                    curr_span
                };

                err.span_suggestion_verbose(
                    span,
                    "associated type for the current `impl` cannot be restricted in `where` \
                     clauses, remove this bound",
                    "",
                    Applicability::MaybeIncorrect,
                );
            }
            if let Some(new) =
                tcx.associated_items(data.impl_or_alias_def_id).find_by_ident_and_kind(
                    tcx,
                    Ident::with_dummy_span(name),
                    ty::AssocTag::Type,
                    data.impl_or_alias_def_id,
                )
            {
                // The associated type is specified in the `impl` we're
                // looking at. Point at it.
                let span = tcx.def_span(new.def_id);
                err.span_label(
                    span,
                    format!(
                        "associated type `<{self_ty_str} as {trait_name}>::{name}` is specified \
                         here",
                    ),
                );
                // Search for the associated type `Self::{name}`, get
                // its type and suggest replacing the bound with it.
                let mut visitor = SelfVisitor { paths: vec![], name: Some(name) };
                visitor.visit_trait_ref(trait_ref);
                for path in visitor.paths {
                    err.span_suggestion_verbose(
                        path.span,
                        "replace the associated type with the type specified in this `impl`",
                        tcx.type_of(new.def_id).skip_binder(),
                        Applicability::MachineApplicable,
                    );
                }
            } else {
                let mut visitor = SelfVisitor { paths: vec![], name: None };
                visitor.visit_trait_ref(trait_ref);
                let span: MultiSpan =
                    visitor.paths.iter().map(|p| p.span).collect::<Vec<Span>>().into();
                err.span_note(
                    span,
                    "associated types for the current `impl` cannot be restricted in `where` \
                     clauses",
                );
            }
        }
        prev = Some((pred, curr_span));
    }
}

fn get_deref_type_and_refs(mut ty: Ty<'_>) -> (Ty<'_>, Vec<hir::Mutability>) {
    let mut refs = vec![];

    while let ty::Ref(_, new_ty, mutbl) = ty.kind() {
        ty = *new_ty;
        refs.push(*mutbl);
    }

    (ty, refs)
}

/// Look for type `param` in an ADT being used only through a reference to confirm that suggesting
/// `param: ?Sized` would be a valid constraint.
struct FindTypeParam {
    param: rustc_span::Symbol,
    invalid_spans: Vec<Span>,
    nested: bool,
}

impl<'v> Visitor<'v> for FindTypeParam {
    fn visit_where_predicate(&mut self, _: &'v hir::WherePredicate<'v>) {
        // Skip where-clauses, to avoid suggesting indirection for type parameters found there.
    }

    fn visit_ty(&mut self, ty: &hir::Ty<'_, AmbigArg>) {
        // We collect the spans of all uses of the "bare" type param, like in `field: T` or
        // `field: (T, T)` where we could make `T: ?Sized` while skipping cases that are known to be
        // valid like `field: &'a T` or `field: *mut T` and cases that *might* have further `Sized`
        // obligations like `Box<T>` and `Vec<T>`, but we perform no extra analysis for those cases
        // and suggest `T: ?Sized` regardless of their obligations. This is fine because the errors
        // in that case should make what happened clear enough.
        match ty.kind {
            hir::TyKind::Ptr(_) | hir::TyKind::Ref(..) | hir::TyKind::TraitObject(..) => {}
            hir::TyKind::Path(hir::QPath::Resolved(None, path))
                if let [segment] = path.segments
                    && segment.ident.name == self.param =>
            {
                if !self.nested {
                    debug!(?ty, "FindTypeParam::visit_ty");
                    self.invalid_spans.push(ty.span);
                }
            }
            hir::TyKind::Path(_) => {
                let prev = self.nested;
                self.nested = true;
                hir::intravisit::walk_ty(self, ty);
                self.nested = prev;
            }
            _ => {
                hir::intravisit::walk_ty(self, ty);
            }
        }
    }
}
