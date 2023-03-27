// ignore-tidy-filelength

use super::{
    DefIdOrName, FindExprBySpan, ImplCandidate, Obligation, ObligationCause, ObligationCauseCode,
    PredicateObligation,
};

use crate::infer::InferCtxt;
use crate::traits::{NormalizeExt, ObligationCtxt};

use hir::def::CtorOf;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::{
    error_code, pluralize, struct_span_err, Applicability, Diagnostic, DiagnosticBuilder,
    ErrorGuaranteed, MultiSpan, Style,
};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::Visitor;
use rustc_hir::is_range_literal;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{AsyncGeneratorKind, GeneratorKind, Node};
use rustc_hir::{Expr, HirId};
use rustc_infer::infer::error_reporting::TypeErrCtxt;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{InferOk, LateBoundRegionConversionTime};
use rustc_middle::hir::map;
use rustc_middle::ty::error::TypeError::{self, Sorts};
use rustc_middle::ty::relate::TypeRelation;
use rustc_middle::ty::{
    self, suggest_arbitrary_trait_bound, suggest_constraining_type_param, AdtKind,
    GeneratorDiagnosticData, GeneratorInteriorTypeCause, Infer, InferTy, InternalSubsts,
    IsSuggestable, ToPredicate, Ty, TyCtxt, TypeAndMut, TypeFoldable, TypeFolder,
    TypeSuperFoldable, TypeVisitableExt, TypeckResults,
};
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::{BytePos, DesugaringKind, ExpnKind, MacroKind, Span, DUMMY_SP};
use rustc_target::spec::abi;
use std::ops::Deref;

use super::method_chain::CollectAllMismatches;
use super::InferCtxtPrivExt;
use crate::infer::InferCtxtExt as _;
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_middle::ty::print::{with_forced_trimmed_paths, with_no_trimmed_paths};

#[derive(Debug)]
pub enum GeneratorInteriorOrUpvar {
    // span of interior type
    Interior(Span, Option<(Option<Span>, Span, Option<hir::HirId>, Option<Span>)>),
    // span of upvar
    Upvar(Span),
}

// This type provides a uniform interface to retrieve data on generators, whether it originated from
// the local crate being compiled or from a foreign crate.
#[derive(Debug)]
pub enum GeneratorData<'tcx, 'a> {
    Local(&'a TypeckResults<'tcx>),
    Foreign(&'tcx GeneratorDiagnosticData<'tcx>),
}

impl<'tcx, 'a> GeneratorData<'tcx, 'a> {
    // Try to get information about variables captured by the generator that matches a type we are
    // looking for with `ty_matches` function. We uses it to find upvar which causes a failure to
    // meet an obligation
    fn try_get_upvar_span<F>(
        &self,
        infer_context: &InferCtxt<'tcx>,
        generator_did: DefId,
        ty_matches: F,
    ) -> Option<GeneratorInteriorOrUpvar>
    where
        F: Fn(ty::Binder<'tcx, Ty<'tcx>>) -> bool,
    {
        match self {
            GeneratorData::Local(typeck_results) => {
                infer_context.tcx.upvars_mentioned(generator_did).and_then(|upvars| {
                    upvars.iter().find_map(|(upvar_id, upvar)| {
                        let upvar_ty = typeck_results.node_type(*upvar_id);
                        let upvar_ty = infer_context.resolve_vars_if_possible(upvar_ty);
                        ty_matches(ty::Binder::dummy(upvar_ty))
                            .then(|| GeneratorInteriorOrUpvar::Upvar(upvar.span))
                    })
                })
            }
            GeneratorData::Foreign(_) => None,
        }
    }

    // Try to get the span of a type being awaited on that matches the type we are looking with the
    // `ty_matches` function. We uses it to find awaited type which causes a failure to meet an
    // obligation
    fn get_from_await_ty<F>(
        &self,
        tcx: TyCtxt<'tcx>,
        visitor: AwaitsVisitor,
        hir: map::Map<'tcx>,
        ty_matches: F,
    ) -> Option<Span>
    where
        F: Fn(ty::Binder<'tcx, Ty<'tcx>>) -> bool,
    {
        match self {
            GeneratorData::Local(typeck_results) => visitor
                .awaits
                .into_iter()
                .map(|id| hir.expect_expr(id))
                .find(|await_expr| {
                    ty_matches(ty::Binder::dummy(typeck_results.expr_ty_adjusted(&await_expr)))
                })
                .map(|expr| expr.span),
            GeneratorData::Foreign(generator_diagnostic_data) => visitor
                .awaits
                .into_iter()
                .map(|id| hir.expect_expr(id))
                .find(|await_expr| {
                    ty_matches(ty::Binder::dummy(
                        generator_diagnostic_data
                            .adjustments
                            .get(&await_expr.hir_id.local_id)
                            .map_or::<&[ty::adjustment::Adjustment<'tcx>], _>(&[], |a| &a[..])
                            .last()
                            .map_or_else::<Ty<'tcx>, _, _>(
                                || {
                                    generator_diagnostic_data
                                        .nodes_types
                                        .get(&await_expr.hir_id.local_id)
                                        .cloned()
                                        .unwrap_or_else(|| {
                                            bug!(
                                                "node_type: no type for node {}",
                                                tcx.hir().node_to_string(await_expr.hir_id)
                                            )
                                        })
                                },
                                |adj| adj.target,
                            ),
                    ))
                })
                .map(|expr| expr.span),
        }
    }

    /// Get the type, expression, span and optional scope span of all types
    /// that are live across the yield of this generator
    fn get_generator_interior_types(
        &self,
    ) -> ty::Binder<'tcx, &[GeneratorInteriorTypeCause<'tcx>]> {
        match self {
            GeneratorData::Local(typeck_result) => {
                typeck_result.generator_interior_types.as_deref()
            }
            GeneratorData::Foreign(generator_diagnostic_data) => {
                generator_diagnostic_data.generator_interior_types.as_deref()
            }
        }
    }

    // Used to get the source of the data, note we don't have as much information for generators
    // originated from foreign crates
    fn is_foreign(&self) -> bool {
        match self {
            GeneratorData::Local(_) => false,
            GeneratorData::Foreign(_) => true,
        }
    }
}

// This trait is public to expose the diagnostics methods to clippy.
pub trait TypeErrCtxtExt<'tcx> {
    fn suggest_restricting_param_bound(
        &self,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        associated_item: Option<(&'static str, Ty<'tcx>)>,
        body_id: LocalDefId,
    );

    fn suggest_dereferences(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool;

    fn get_closure_name(&self, def_id: DefId, err: &mut Diagnostic, msg: &str) -> Option<Symbol>;

    fn suggest_fn_call(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool;

    fn check_for_binding_assigned_block_without_tail_expression(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    );

    fn suggest_add_clone_to_arg(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool;

    fn extract_callable_info(
        &self,
        body_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        found: Ty<'tcx>,
    ) -> Option<(DefIdOrName, Ty<'tcx>, Vec<Ty<'tcx>>)>;

    fn suggest_add_reference_to_arg(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        has_custom_message: bool,
    ) -> bool;

    fn suggest_borrowing_for_object_cast(
        &self,
        err: &mut Diagnostic,
        obligation: &PredicateObligation<'tcx>,
        self_ty: Ty<'tcx>,
        object_ty: Ty<'tcx>,
    );

    fn suggest_remove_reference(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool;

    fn suggest_remove_await(&self, obligation: &PredicateObligation<'tcx>, err: &mut Diagnostic);

    fn suggest_change_mut(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    );

    fn suggest_semicolon_removal(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        span: Span,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool;

    fn return_type_span(&self, obligation: &PredicateObligation<'tcx>) -> Option<Span>;

    fn suggest_impl_trait(
        &self,
        err: &mut Diagnostic,
        span: Span,
        obligation: &PredicateObligation<'tcx>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool;

    fn point_at_returns_when_relevant(
        &self,
        err: &mut DiagnosticBuilder<'tcx, ErrorGuaranteed>,
        obligation: &PredicateObligation<'tcx>,
    );

    fn report_closure_arg_mismatch(
        &self,
        span: Span,
        found_span: Option<Span>,
        found: ty::PolyTraitRef<'tcx>,
        expected: ty::PolyTraitRef<'tcx>,
        cause: &ObligationCauseCode<'tcx>,
        found_node: Option<Node<'_>>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed>;

    fn note_conflicting_closure_bounds(
        &self,
        cause: &ObligationCauseCode<'tcx>,
        err: &mut DiagnosticBuilder<'tcx, ErrorGuaranteed>,
    );

    fn suggest_fully_qualified_path(
        &self,
        err: &mut Diagnostic,
        item_def_id: DefId,
        span: Span,
        trait_ref: DefId,
    );

    fn maybe_note_obligation_cause_for_async_await(
        &self,
        err: &mut Diagnostic,
        obligation: &PredicateObligation<'tcx>,
    ) -> bool;

    fn note_obligation_cause_for_async_await(
        &self,
        err: &mut Diagnostic,
        interior_or_upvar_span: GeneratorInteriorOrUpvar,
        is_async: bool,
        outer_generator: Option<DefId>,
        trait_pred: ty::TraitPredicate<'tcx>,
        target_ty: Ty<'tcx>,
        typeck_results: Option<&ty::TypeckResults<'tcx>>,
        obligation: &PredicateObligation<'tcx>,
        next_code: Option<&ObligationCauseCode<'tcx>>,
    );

    fn note_obligation_cause_code<T>(
        &self,
        err: &mut Diagnostic,
        predicate: T,
        param_env: ty::ParamEnv<'tcx>,
        cause_code: &ObligationCauseCode<'tcx>,
        obligated_types: &mut Vec<Ty<'tcx>>,
        seen_requirements: &mut FxHashSet<DefId>,
    ) where
        T: ToPredicate<'tcx>;

    /// Suggest to await before try: future? => future.await?
    fn suggest_await_before_try(
        &self,
        err: &mut Diagnostic,
        obligation: &PredicateObligation<'tcx>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        span: Span,
    );

    fn suggest_floating_point_literal(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_ref: &ty::PolyTraitRef<'tcx>,
    );

    fn suggest_derive(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    );

    fn suggest_dereferencing_index(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    );
    fn note_function_argument_obligation(
        &self,
        arg_hir_id: HirId,
        err: &mut Diagnostic,
        parent_code: &ObligationCauseCode<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        predicate: ty::Predicate<'tcx>,
        call_hir_id: HirId,
    );
    fn point_at_chain(
        &self,
        expr: &hir::Expr<'_>,
        typeck_results: &TypeckResults<'tcx>,
        type_diffs: Vec<TypeError<'tcx>>,
        param_env: ty::ParamEnv<'tcx>,
        err: &mut Diagnostic,
    );
    fn probe_assoc_types_at_expr(
        &self,
        type_diffs: &[TypeError<'tcx>],
        span: Span,
        prev_ty: Ty<'tcx>,
        body_id: hir::HirId,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Vec<Option<(Span, (DefId, Ty<'tcx>))>>;

    fn maybe_suggest_convert_to_slice(
        &self,
        err: &mut Diagnostic,
        trait_ref: ty::Binder<'tcx, ty::TraitRef<'tcx>>,
        candidate_impls: &[ImplCandidate<'tcx>],
        span: Span,
    );
}

fn predicate_constraint(generics: &hir::Generics<'_>, pred: ty::Predicate<'_>) -> (Span, String) {
    (
        generics.tail_span_for_predicate_suggestion(),
        format!("{} {}", generics.add_where_or_trailing_comma(), pred),
    )
}

/// Type parameter needs more bounds. The trivial case is `T` `where T: Bound`, but
/// it can also be an `impl Trait` param that needs to be decomposed to a type
/// param for cleaner code.
fn suggest_restriction<'tcx>(
    tcx: TyCtxt<'tcx>,
    item_id: LocalDefId,
    hir_generics: &hir::Generics<'tcx>,
    msg: &str,
    err: &mut Diagnostic,
    fn_sig: Option<&hir::FnSig<'_>>,
    projection: Option<&ty::AliasTy<'_>>,
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
        || projection.map_or(false, |projection| tcx.opt_rpitit_info(projection.def_id).is_some())
    {
        return;
    }
    let generics = tcx.generics_of(item_id);
    // Given `fn foo(t: impl Trait)` where `Trait` requires assoc type `A`...
    if let Some((param, bound_str, fn_sig)) =
        fn_sig.zip(projection).and_then(|(sig, p)| match p.self_ty().kind() {
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
                .visit_ty(input);
        }
        // The type param `T: Trait` we will suggest to introduce.
        let type_param = format!("{}: {}", type_param_name, bound_str);

        let mut sugg = vec![
            if let Some(span) = hir_generics.span_for_param_suggestion() {
                (span, format!(", {}", type_param))
            } else {
                (hir_generics.span, format!("<{}>", type_param))
            },
            // `fn foo(t: impl Trait)`
            //                       ^ suggest `where <T as Trait>::A: Bound`
            predicate_constraint(hir_generics, trait_pred.to_predicate(tcx)),
        ];
        sugg.extend(ty_spans.into_iter().map(|s| (s, type_param_name.to_string())));

        // Suggest `fn foo<T: Trait>(t: T) where <T as Trait>::A: Bound`.
        // FIXME: once `#![feature(associated_type_bounds)]` is stabilized, we should suggest
        // `fn foo(t: impl Trait<A: Bound>)` instead.
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
            (_, None) => predicate_constraint(hir_generics, trait_pred.to_predicate(tcx)),
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
            &format!("consider further restricting {}", msg),
            suggestion,
            Applicability::MachineApplicable,
        );
    }
}

impl<'tcx> TypeErrCtxtExt<'tcx> for TypeErrCtxt<'_, 'tcx> {
    fn suggest_restricting_param_bound(
        &self,
        mut err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        associated_ty: Option<(&'static str, Ty<'tcx>)>,
        mut body_id: LocalDefId,
    ) {
        let trait_pred = self.resolve_numeric_literals_with_default(trait_pred);

        let self_ty = trait_pred.skip_binder().self_ty();
        let (param_ty, projection) = match self_ty.kind() {
            ty::Param(_) => (true, None),
            ty::Alias(ty::Projection, projection) => (false, Some(projection)),
            _ => (false, None),
        };

        // FIXME: Add check for trait bound that is already present, particularly `?Sized` so we
        //        don't suggest `T: Sized + ?Sized`.
        while let Some(node) = self.tcx.hir().find_by_def_id(body_id) {
            match node {
                hir::Node::Item(hir::Item {
                    ident,
                    kind: hir::ItemKind::Trait(_, _, generics, bounds, _),
                    ..
                }) if self_ty == self.tcx.types.self_param => {
                    assert!(param_ty);
                    // Restricting `Self` for a single method.
                    suggest_restriction(
                        self.tcx,
                        body_id,
                        &generics,
                        "`Self`",
                        err,
                        None,
                        projection,
                        trait_pred,
                        Some((ident, bounds)),
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
                        self.tcx, body_id, &generics, "`Self`", err, None, projection, trait_pred,
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
                    kind: hir::ItemKind::Fn(fn_sig, generics, _), ..
                }) if projection.is_some() => {
                    // Missing restriction on associated type of type parameter (unmet projection).
                    suggest_restriction(
                        self.tcx,
                        body_id,
                        &generics,
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
                        hir::ItemKind::Trait(_, _, generics, ..)
                        | hir::ItemKind::Impl(hir::Impl { generics, .. }),
                    ..
                }) if projection.is_some() => {
                    // Missing restriction on associated type of type parameter (unmet projection).
                    suggest_restriction(
                        self.tcx,
                        body_id,
                        &generics,
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
                        hir::ItemKind::Struct(_, generics)
                        | hir::ItemKind::Enum(_, generics)
                        | hir::ItemKind::Union(_, generics)
                        | hir::ItemKind::Trait(_, _, generics, ..)
                        | hir::ItemKind::Impl(hir::Impl { generics, .. })
                        | hir::ItemKind::Fn(_, generics, _)
                        | hir::ItemKind::TyAlias(_, generics)
                        | hir::ItemKind::TraitAlias(generics, _)
                        | hir::ItemKind::OpaqueTy(hir::OpaqueTy { generics, .. }),
                    ..
                })
                | hir::Node::TraitItem(hir::TraitItem { generics, .. })
                | hir::Node::ImplItem(hir::ImplItem { generics, .. })
                    if param_ty =>
                {
                    // We skip the 0'th subst (self) because we do not want
                    // to consider the predicate as not suggestible if the
                    // self type is an arg position `impl Trait` -- instead,
                    // we handle that by adding ` + Bound` below.
                    // FIXME(compiler-errors): It would be nice to do the same
                    // this that we do in `suggest_restriction` and pull the
                    // `impl Trait` into a new generic if it shows up somewhere
                    // else in the predicate.
                    if !trait_pred.skip_binder().trait_ref.substs[1..]
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
                        if constraint.ends_with('>') {
                            constraint = format!(
                                "{}, {} = {}>",
                                &constraint[..constraint.len() - 1],
                                name,
                                term
                            );
                        } else {
                            constraint.push_str(&format!("<{} = {}>", name, term));
                        }
                    }

                    if suggest_constraining_type_param(
                        self.tcx,
                        generics,
                        &mut err,
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
                        hir::ItemKind::Struct(_, generics)
                        | hir::ItemKind::Enum(_, generics)
                        | hir::ItemKind::Union(_, generics)
                        | hir::ItemKind::Trait(_, _, generics, ..)
                        | hir::ItemKind::Impl(hir::Impl { generics, .. })
                        | hir::ItemKind::Fn(_, generics, _)
                        | hir::ItemKind::TyAlias(_, generics)
                        | hir::ItemKind::TraitAlias(generics, _)
                        | hir::ItemKind::OpaqueTy(hir::OpaqueTy { generics, .. }),
                    ..
                }) if !param_ty => {
                    // Missing generic type parameter bound.
                    if suggest_arbitrary_trait_bound(
                        self.tcx,
                        generics,
                        &mut err,
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

    /// When after several dereferencing, the reference satisfies the trait
    /// binding. This function provides dereference suggestion for this
    /// specific situation.
    fn suggest_dereferences(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        // It only make sense when suggesting dereferences for arguments
        let ObligationCauseCode::FunctionArgumentObligation { arg_hir_id, call_hir_id, .. } = obligation.cause.code()
            else { return false; };
        let Some(typeck_results) = &self.typeck_results
            else { return false; };
        let hir::Node::Expr(expr) = self.tcx.hir().get(*arg_hir_id)
            else { return false; };
        let Some(arg_ty) = typeck_results.expr_ty_adjusted_opt(expr)
            else { return false; };

        let span = obligation.cause.span;
        let mut real_trait_pred = trait_pred;
        let mut code = obligation.cause.code();
        while let Some((parent_code, parent_trait_pred)) = code.parent() {
            code = parent_code;
            if let Some(parent_trait_pred) = parent_trait_pred {
                real_trait_pred = parent_trait_pred;
            }

            let real_ty = real_trait_pred.self_ty();
            // We `erase_late_bound_regions` here because `make_subregion` does not handle
            // `ReLateBound`, and we don't particularly care about the regions.
            if !self.can_eq(
                obligation.param_env,
                self.tcx.erase_late_bound_regions(real_ty),
                arg_ty,
            ) {
                continue;
            }

            if let ty::Ref(region, base_ty, mutbl) = *real_ty.skip_binder().kind() {
                let autoderef = (self.autoderef_steps)(base_ty);
                if let Some(steps) =
                    autoderef.into_iter().enumerate().find_map(|(steps, (ty, obligations))| {
                        // Re-add the `&`
                        let ty = self.tcx.mk_ref(region, TypeAndMut { ty, mutbl });

                        // Remapping bound vars here
                        let real_trait_pred_and_ty =
                            real_trait_pred.map_bound(|inner_trait_pred| (inner_trait_pred, ty));
                        let obligation = self.mk_trait_obligation_with_new_self_ty(
                            obligation.param_env,
                            real_trait_pred_and_ty,
                        );
                        let may_hold = obligations
                            .iter()
                            .chain([&obligation])
                            .all(|obligation| self.predicate_may_hold(obligation))
                            .then_some(steps);

                        may_hold
                    })
                {
                    if steps > 0 {
                        // Don't care about `&mut` because `DerefMut` is used less
                        // often and user will not expect autoderef happens.
                        if let Some(hir::Node::Expr(hir::Expr {
                            kind:
                                hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, expr),
                            ..
                        })) = self.tcx.hir().find(*arg_hir_id)
                        {
                            let derefs = "*".repeat(steps);
                            err.span_suggestion_verbose(
                                expr.span.shrink_to_lo(),
                                "consider dereferencing here",
                                derefs,
                                Applicability::MachineApplicable,
                            );
                            return true;
                        }
                    }
                } else if real_trait_pred != trait_pred {
                    // This branch addresses #87437.

                    // Remapping bound vars here
                    let real_trait_pred_and_base_ty =
                        real_trait_pred.map_bound(|inner_trait_pred| (inner_trait_pred, base_ty));
                    let obligation = self.mk_trait_obligation_with_new_self_ty(
                        obligation.param_env,
                        real_trait_pred_and_base_ty,
                    );
                    if self.predicate_may_hold(&obligation) {
                        let call_node = self.tcx.hir().get(*call_hir_id);
                        let msg = "consider dereferencing here";
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
                                    (span.shrink_to_lo(), "(*".to_string()),
                                    (span.shrink_to_hi(), ")".to_string()),
                                ],
                                Applicability::MachineApplicable,
                            )
                        } else {
                            err.span_suggestion_verbose(
                                span.shrink_to_lo(),
                                msg,
                                '*',
                                Applicability::MachineApplicable,
                            )
                        };
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Given a closure's `DefId`, return the given name of the closure.
    ///
    /// This doesn't account for reassignments, but it's only used for suggestions.
    fn get_closure_name(&self, def_id: DefId, err: &mut Diagnostic, msg: &str) -> Option<Symbol> {
        let get_name = |err: &mut Diagnostic, kind: &hir::PatKind<'_>| -> Option<Symbol> {
            // Get the local name of this closure. This can be inaccurate because
            // of the possibility of reassignment, but this should be good enough.
            match &kind {
                hir::PatKind::Binding(hir::BindingAnnotation::NONE, _, ident, None) => {
                    Some(ident.name)
                }
                _ => {
                    err.note(msg);
                    None
                }
            }
        };

        let hir = self.tcx.hir();
        let hir_id = hir.local_def_id_to_hir_id(def_id.as_local()?);
        match hir.find_parent(hir_id) {
            Some(hir::Node::Stmt(hir::Stmt { kind: hir::StmtKind::Local(local), .. })) => {
                get_name(err, &local.pat.kind)
            }
            // Different to previous arm because one is `&hir::Local` and the other
            // is `P<hir::Local>`.
            Some(hir::Node::Local(local)) => get_name(err, &local.pat.kind),
            _ => None,
        }
    }

    /// We tried to apply the bound to an `fn` or closure. Check whether calling it would
    /// evaluate to a type that *would* satisfy the trait binding. If it would, suggest calling
    /// it: `bar(foo)` → `bar(foo())`. This case is *very* likely to be hit if `foo` is `async`.
    fn suggest_fn_call(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        // It doesn't make sense to make this suggestion outside of typeck...
        // (also autoderef will ICE...)
        if self.typeck_results.is_none() {
            return false;
        }

        if let ty::PredicateKind::Clause(ty::Clause::Trait(trait_pred)) = obligation.predicate.kind().skip_binder()
            && Some(trait_pred.def_id()) == self.tcx.lang_items().sized_trait()
        {
            // Don't suggest calling to turn an unsized type into a sized type
            return false;
        }

        let self_ty = self.instantiate_binder_with_fresh_vars(
            DUMMY_SP,
            LateBoundRegionConversionTime::FnCall,
            trait_pred.self_ty(),
        );

        let Some((def_id_or_name, output, inputs)) = self.extract_callable_info(
            obligation.cause.body_id,
            obligation.param_env,
            self_ty,
        ) else { return false; };

        // Remapping bound vars here
        let trait_pred_and_self = trait_pred.map_bound(|trait_pred| (trait_pred, output));

        let new_obligation =
            self.mk_trait_obligation_with_new_self_ty(obligation.param_env, trait_pred_and_self);
        if !self.predicate_must_hold_modulo_regions(&new_obligation) {
            return false;
        }

        // Get the name of the callable and the arguments to be used in the suggestion.
        let hir = self.tcx.hir();

        let msg = match def_id_or_name {
            DefIdOrName::DefId(def_id) => match self.tcx.def_kind(def_id) {
                DefKind::Ctor(CtorOf::Struct, _) => {
                    "use parentheses to construct this tuple struct".to_string()
                }
                DefKind::Ctor(CtorOf::Variant, _) => {
                    "use parentheses to construct this tuple variant".to_string()
                }
                kind => format!(
                    "use parentheses to call this {}",
                    self.tcx.def_kind_descr(kind, def_id)
                ),
            },
            DefIdOrName::Name(name) => format!("use parentheses to call this {name}"),
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

        if matches!(obligation.cause.code(), ObligationCauseCode::FunctionArgumentObligation { .. })
            && obligation.cause.span.can_be_used_for_suggestions()
        {
            // When the obligation error has been ensured to have been caused by
            // an argument, the `obligation.cause.span` points at the expression
            // of the argument, so we can provide a suggestion. Otherwise, we give
            // a more general note.
            err.span_suggestion_verbose(
                obligation.cause.span.shrink_to_hi(),
                &msg,
                format!("({args})"),
                Applicability::HasPlaceholders,
            );
        } else if let DefIdOrName::DefId(def_id) = def_id_or_name {
            let name = match hir.get_if_local(def_id) {
                Some(hir::Node::Expr(hir::Expr {
                    kind: hir::ExprKind::Closure(hir::Closure { fn_decl_span, .. }),
                    ..
                })) => {
                    err.span_label(*fn_decl_span, "consider calling this closure");
                    let Some(name) = self.get_closure_name(def_id, err, &msg) else {
                        return false;
                    };
                    name.to_string()
                }
                Some(hir::Node::Item(hir::Item { ident, kind: hir::ItemKind::Fn(..), .. })) => {
                    err.span_label(ident.span, "consider calling this function");
                    ident.to_string()
                }
                Some(hir::Node::Ctor(..)) => {
                    let name = self.tcx.def_path_str(def_id);
                    err.span_label(
                        self.tcx.def_span(def_id),
                        format!("consider calling the constructor for `{}`", name),
                    );
                    name
                }
                _ => return false,
            };
            err.help(&format!("{msg}: `{name}({args})`"));
        }
        true
    }

    fn check_for_binding_assigned_block_without_tail_expression(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) {
        let mut span = obligation.cause.span;
        while span.from_expansion() {
            // Remove all the desugaring and macro contexts.
            span.remove_mark();
        }
        let mut expr_finder = FindExprBySpan::new(span);
        let Some(body_id) = self.tcx.hir().maybe_body_owned_by(obligation.cause.body_id) else { return; };
        let body = self.tcx.hir().body(body_id);
        expr_finder.visit_expr(body.value);
        let Some(expr) = expr_finder.result else { return; };
        let Some(typeck) = &self.typeck_results else { return; };
        let Some(ty) = typeck.expr_ty_adjusted_opt(expr) else { return; };
        if !ty.is_unit() {
            return;
        };
        let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = expr.kind else { return; };
        let hir::def::Res::Local(hir_id) = path.res else { return; };
        let Some(hir::Node::Pat(pat)) = self.tcx.hir().find(hir_id) else {
            return;
        };
        let Some(hir::Node::Local(hir::Local {
            ty: None,
            init: Some(init),
            ..
        })) = self.tcx.hir().find_parent(pat.hir_id) else { return; };
        let hir::ExprKind::Block(block, None) = init.kind else { return; };
        if block.expr.is_some() {
            return;
        }
        let [.., stmt] = block.stmts else {
            err.span_label(block.span, "this empty block is missing a tail expression");
            return;
        };
        let hir::StmtKind::Semi(tail_expr) = stmt.kind else { return; };
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

    fn suggest_add_clone_to_arg(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        let self_ty = self.resolve_vars_if_possible(trait_pred.self_ty());
        let ty = self.instantiate_binder_with_placeholders(self_ty);
        let Some(generics) = self.tcx.hir().get_generics(obligation.cause.body_id) else { return false };
        let ty::Ref(_, inner_ty, hir::Mutability::Not) = ty.kind() else { return false };
        let ty::Param(param) = inner_ty.kind() else { return false };
        let ObligationCauseCode::FunctionArgumentObligation { arg_hir_id, .. } = obligation.cause.code() else { return false };
        let arg_node = self.tcx.hir().get(*arg_hir_id);
        let Node::Expr(Expr { kind: hir::ExprKind::Path(_), ..}) = arg_node else { return false };

        let clone_trait = self.tcx.require_lang_item(LangItem::Clone, None);
        let has_clone = |ty| {
            self.type_implements_trait(clone_trait, [ty], obligation.param_env)
                .must_apply_modulo_regions()
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
            err.span_suggestion_verbose(
                obligation.cause.span.shrink_to_hi(),
                "consider using clone here",
                ".clone()".to_string(),
                Applicability::MaybeIncorrect,
            );
            return true;
        }
        false
    }

    /// Extracts information about a callable type for diagnostics. This is a
    /// heuristic -- it doesn't necessarily mean that a type is always callable,
    /// because the callable type must also be well-formed to be called.
    fn extract_callable_info(
        &self,
        body_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        found: Ty<'tcx>,
    ) -> Option<(DefIdOrName, Ty<'tcx>, Vec<Ty<'tcx>>)> {
        // Autoderef is useful here because sometimes we box callables, etc.
        let Some((def_id_or_name, output, inputs)) = (self.autoderef_steps)(found).into_iter().find_map(|(found, _)| {
            match *found.kind() {
                ty::FnPtr(fn_sig) =>
                    Some((DefIdOrName::Name("function pointer"), fn_sig.output(), fn_sig.inputs())),
                ty::FnDef(def_id, _) => {
                    let fn_sig = found.fn_sig(self.tcx);
                    Some((DefIdOrName::DefId(def_id), fn_sig.output(), fn_sig.inputs()))
                }
                ty::Closure(def_id, substs) => {
                    let fn_sig = substs.as_closure().sig();
                    Some((DefIdOrName::DefId(def_id), fn_sig.output(), fn_sig.inputs().map_bound(|inputs| &inputs[1..])))
                }
                ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) => {
                    self.tcx.item_bounds(def_id).subst(self.tcx, substs).iter().find_map(|pred| {
                        if let ty::PredicateKind::Clause(ty::Clause::Projection(proj)) = pred.kind().skip_binder()
                        && Some(proj.projection_ty.def_id) == self.tcx.lang_items().fn_once_output()
                        // args tuple will always be substs[1]
                        && let ty::Tuple(args) = proj.projection_ty.substs.type_at(1).kind()
                        {
                            Some((
                                DefIdOrName::DefId(def_id),
                                pred.kind().rebind(proj.term.ty().unwrap()),
                                pred.kind().rebind(args.as_slice()),
                            ))
                        } else {
                            None
                        }
                    })
                }
                ty::Dynamic(data, _, ty::Dyn) => {
                    data.iter().find_map(|pred| {
                        if let ty::ExistentialPredicate::Projection(proj) = pred.skip_binder()
                        && Some(proj.def_id) == self.tcx.lang_items().fn_once_output()
                        // for existential projection, substs are shifted over by 1
                        && let ty::Tuple(args) = proj.substs.type_at(0).kind()
                        {
                            Some((
                                DefIdOrName::Name("trait object"),
                                pred.rebind(proj.term.ty().unwrap()),
                                pred.rebind(args.as_slice()),
                            ))
                        } else {
                            None
                        }
                    })
                }
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
                        if let ty::PredicateKind::Clause(ty::Clause::Projection(proj)) = pred.kind().skip_binder()
                        && Some(proj.projection_ty.def_id) == self.tcx.lang_items().fn_once_output()
                        && proj.projection_ty.self_ty() == found
                        // args tuple will always be substs[1]
                        && let ty::Tuple(args) = proj.projection_ty.substs.type_at(1).kind()
                        {
                            Some((
                                name,
                                pred.kind().rebind(proj.term.ty().unwrap()),
                                pred.kind().rebind(args.as_slice()),
                            ))
                        } else {
                            None
                        }
                    })
                }
                _ => None,
            }
        }) else { return None; };

        let output = self.instantiate_binder_with_fresh_vars(
            DUMMY_SP,
            LateBoundRegionConversionTime::FnCall,
            output,
        );
        let inputs = inputs
            .skip_binder()
            .iter()
            .map(|ty| {
                self.instantiate_binder_with_fresh_vars(
                    DUMMY_SP,
                    LateBoundRegionConversionTime::FnCall,
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

    fn suggest_add_reference_to_arg(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        poly_trait_pred: ty::PolyTraitPredicate<'tcx>,
        has_custom_message: bool,
    ) -> bool {
        let span = obligation.cause.span;

        let code = if let ObligationCauseCode::FunctionArgumentObligation { parent_code, .. } =
            obligation.cause.code()
        {
            &parent_code
        } else if let ObligationCauseCode::ItemObligation(_)
        | ObligationCauseCode::ExprItemObligation(..) = obligation.cause.code()
        {
            obligation.cause.code()
        } else if let ExpnKind::Desugaring(DesugaringKind::ForLoop) =
            span.ctxt().outer_expn_data().kind
        {
            obligation.cause.code()
        } else {
            return false;
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

        // Try to apply the original trait binding obligation by borrowing.
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
                    self.tcx.mk_imm_ref(self.tcx.lifetimes.re_static, trait_pred.self_ty()),
                )
            });
            let trait_pred_and_mut_ref = old_pred.map_bound(|trait_pred| {
                (
                    trait_pred,
                    self.tcx.mk_mut_ref(self.tcx.lifetimes.re_static, trait_pred.self_ty()),
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
                if let ObligationCauseCode::ItemObligation(_) | ObligationCauseCode::ExprItemObligation(..) = obligation.cause.code()
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

                    let msg = format!("the trait bound `{}` is not satisfied", old_pred);
                    if has_custom_message {
                        err.note(&msg);
                    } else {
                        err.message =
                            vec![(rustc_errors::DiagnosticMessage::Str(msg), Style::NoStyle)];
                    }
                    err.span_label(
                        span,
                        format!(
                            "the trait `{}` is not implemented for `{}`",
                            old_pred.print_modifiers_and_trait_path(),
                            old_pred.self_ty().skip_binder(),
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
                        let sugg_msg = &format!(
                            "consider{} borrowing here",
                            if is_mut { " mutably" } else { "" }
                        );

                        // Issue #109436, we need to add parentheses properly for method calls
                        // for example, `foo.into()` should be `(&foo).into()`
                        if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(
                            self.tcx.sess.source_map().span_look_ahead(span, Some("."), Some(50)),
                        ) {
                            if snippet == "." {
                                err.multipart_suggestion_verbose(
                                    sugg_msg,
                                    vec![
                                        (span.shrink_to_lo(), format!("({}", sugg_prefix)),
                                        (span.shrink_to_hi(), ")".to_string()),
                                    ],
                                    Applicability::MaybeIncorrect,
                                );
                                return true;
                            }
                        }

                        // Issue #104961, we need to add parentheses properly for compond expressions
                        // for example, `x.starts_with("hi".to_string() + "you")`
                        // should be `x.starts_with(&("hi".to_string() + "you"))`
                        let Some(body_id) = self.tcx.hir().maybe_body_owned_by(obligation.cause.body_id) else { return false; };
                        let body = self.tcx.hir().body(body_id);
                        let mut expr_finder = FindExprBySpan::new(span);
                        expr_finder.visit_expr(body.value);
                        let Some(expr) = expr_finder.result else { return false; };
                        let needs_parens = match expr.kind {
                            // parenthesize if needed (Issue #46756)
                            hir::ExprKind::Cast(_, _) | hir::ExprKind::Binary(_, _, _) => true,
                            // parenthesize borrows of range literals (Issue #54505)
                            _ if is_range_literal(expr) => true,
                            _ => false,
                        };

                        let span = if needs_parens { span } else { span.shrink_to_lo() };
                        let suggestions = if !needs_parens {
                            vec![(span.shrink_to_lo(), format!("{}", sugg_prefix))]
                        } else {
                            vec![
                                (span.shrink_to_lo(), format!("{}(", sugg_prefix)),
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

        if let ObligationCauseCode::ImplDerivedObligation(cause) = &*code {
            try_borrowing(cause.derived.parent_trait_pred, &[])
        } else if let ObligationCauseCode::BindingObligation(_, _)
        | ObligationCauseCode::ItemObligation(_)
        | ObligationCauseCode::ExprItemObligation(..)
        | ObligationCauseCode::ExprBindingObligation(..) = code
        {
            try_borrowing(poly_trait_pred, &never_suggest_borrow)
        } else {
            false
        }
    }

    // Suggest borrowing the type
    fn suggest_borrowing_for_object_cast(
        &self,
        err: &mut Diagnostic,
        obligation: &PredicateObligation<'tcx>,
        self_ty: Ty<'tcx>,
        object_ty: Ty<'tcx>,
    ) {
        let ty::Dynamic(predicates, _, ty::Dyn) = object_ty.kind() else { return; };
        let self_ref_ty = self.tcx.mk_imm_ref(self.tcx.lifetimes.re_erased, self_ty);

        for predicate in predicates.iter() {
            if !self.predicate_must_hold_modulo_regions(
                &obligation.with(self.tcx, predicate.with_self_ty(self.tcx, self_ref_ty)),
            ) {
                return;
            }
        }

        err.span_suggestion(
            obligation.cause.span.shrink_to_lo(),
            &format!(
                "consider borrowing the value, since `&{self_ty}` can be coerced into `{object_ty}`"
            ),
            "&",
            Applicability::MaybeIncorrect,
        );
    }

    /// Whenever references are used by mistake, like `for (i, e) in &vec.iter().enumerate()`,
    /// suggest removing these references until we reach a type that implements the trait.
    fn suggest_remove_reference(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        let mut span = obligation.cause.span;
        let mut trait_pred = trait_pred;
        let mut code = obligation.cause.code();
        while let Some((c, Some(parent_trait_pred))) = code.parent() {
            // We want the root obligation, in order to detect properly handle
            // `for _ in &mut &mut vec![] {}`.
            code = c;
            trait_pred = parent_trait_pred;
        }
        while span.desugaring_kind().is_some() {
            // Remove all the hir desugaring contexts while maintaining the macro contexts.
            span.remove_mark();
        }
        let mut expr_finder = super::FindExprBySpan::new(span);
        let Some(body_id) = self.tcx.hir().maybe_body_owned_by(obligation.cause.body_id) else {
            return false;
        };
        let body = self.tcx.hir().body(body_id);
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
                    &msg,
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
        let Some(mut expr) = expr_finder.result else { return false; };
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
                && let hir::def::Res::Local(hir_id) = path.res
                && let Some(hir::Node::Pat(binding)) = self.tcx.hir().find(hir_id)
                && let Some(hir::Node::Local(local)) = self.tcx.hir().find_parent(binding.hir_id)
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

    fn suggest_remove_await(&self, obligation: &PredicateObligation<'tcx>, err: &mut Diagnostic) {
        let span = obligation.cause.span;

        if let ObligationCauseCode::AwaitableExpr(hir_id) = obligation.cause.code().peel_derives() {
            let hir = self.tcx.hir();
            if let Some(hir::Node::Expr(expr)) = hir_id.and_then(|hir_id| hir.find(hir_id)) {
                // FIXME: use `obligation.predicate.kind()...trait_ref.self_ty()` to see if we have `()`
                // and if not maybe suggest doing something else? If we kept the expression around we
                // could also check if it is an fn call (very likely) and suggest changing *that*, if
                // it is from the local crate.
                err.span_suggestion(
                    span,
                    "remove the `.await`",
                    "",
                    Applicability::MachineApplicable,
                );
                // FIXME: account for associated `async fn`s.
                if let hir::Expr { span, kind: hir::ExprKind::Call(base, _), .. } = expr {
                    if let ty::PredicateKind::Clause(ty::Clause::Trait(pred)) =
                        obligation.predicate.kind().skip_binder()
                    {
                        err.span_label(*span, &format!("this call returns `{}`", pred.self_ty()));
                    }
                    if let Some(typeck_results) = &self.typeck_results
                            && let ty = typeck_results.expr_ty_adjusted(base)
                            && let ty::FnDef(def_id, _substs) = ty.kind()
                            && let Some(hir::Node::Item(hir::Item { ident, span, vis_span, .. })) =
                                hir.get_if_local(*def_id)
                        {
                            let msg = format!(
                                "alternatively, consider making `fn {}` asynchronous",
                                ident
                            );
                            if vis_span.is_empty() {
                                err.span_suggestion_verbose(
                                    span.shrink_to_lo(),
                                    &msg,
                                    "async ",
                                    Applicability::MaybeIncorrect,
                                );
                            } else {
                                err.span_suggestion_verbose(
                                    vis_span.shrink_to_hi(),
                                    &msg,
                                    " async",
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        }
                }
            }
        }
    }

    /// Check if the trait bound is implemented for a different mutability and note it in the
    /// final error.
    fn suggest_change_mut(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) {
        let points_at_arg = matches!(
            obligation.cause.code(),
            ObligationCauseCode::FunctionArgumentObligation { .. },
        );

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
                    hir::Mutability::Mut => self.tcx.mk_imm_ref(region, t_type),
                    hir::Mutability::Not => self.tcx.mk_mut_ref(region, t_type),
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
                        err.note(&format!(
                            "`{}` is implemented for `{:?}`, but not for `{:?}`",
                            trait_pred.print_modifiers_and_trait_path(),
                            suggested_ty,
                            trait_pred.skip_binder().self_ty(),
                        ));
                    }
                }
            }
        }
    }

    fn suggest_semicolon_removal(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        span: Span,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        let hir = self.tcx.hir();
        let node = hir.find_by_def_id(obligation.cause.body_id);
        if let Some(hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(sig, _, body_id), .. })) = node
            && let hir::ExprKind::Block(blk, _) = &hir.body(*body_id).value.kind
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
                &format!(
                    "this expression has type `{}`, which implements `{}`",
                    ty,
                    trait_pred.print_modifiers_and_trait_path()
                )
            );
            err.span_suggestion(
                self.tcx.sess.source_map().end_point(stmt.span),
                "remove this semicolon",
                "",
                Applicability::MachineApplicable
            );
            return true;
        }
        false
    }

    fn return_type_span(&self, obligation: &PredicateObligation<'tcx>) -> Option<Span> {
        let hir = self.tcx.hir();
        let Some(hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(sig, ..), .. })) = hir.find_by_def_id(obligation.cause.body_id) else {
            return None;
        };

        if let hir::FnRetTy::Return(ret_ty) = sig.decl.output { Some(ret_ty.span) } else { None }
    }

    /// If all conditions are met to identify a returned `dyn Trait`, suggest using `impl Trait` if
    /// applicable and signal that the error has been expanded appropriately and needs to be
    /// emitted.
    fn suggest_impl_trait(
        &self,
        err: &mut Diagnostic,
        span: Span,
        obligation: &PredicateObligation<'tcx>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) -> bool {
        match obligation.cause.code().peel_derives() {
            // Only suggest `impl Trait` if the return type is unsized because it is `dyn Trait`.
            ObligationCauseCode::SizedReturnType => {}
            _ => return false,
        }

        let hir = self.tcx.hir();
        let fn_hir_id = hir.local_def_id_to_hir_id(obligation.cause.body_id);
        let node = hir.find_by_def_id(obligation.cause.body_id);
        let Some(hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Fn(sig, _, body_id),
            ..
        })) = node
        else {
            return false;
        };
        let body = hir.body(*body_id);
        let trait_pred = self.resolve_vars_if_possible(trait_pred);
        let ty = trait_pred.skip_binder().self_ty();
        let is_object_safe = match ty.kind() {
            ty::Dynamic(predicates, _, ty::Dyn) => {
                // If the `dyn Trait` is not object safe, do not suggest `Box<dyn Trait>`.
                predicates
                    .principal_def_id()
                    .map_or(true, |def_id| self.tcx.check_is_object_safe(def_id))
            }
            // We only want to suggest `impl Trait` to `dyn Trait`s.
            // For example, `fn foo() -> str` needs to be filtered out.
            _ => return false,
        };

        let hir::FnRetTy::Return(ret_ty) = sig.decl.output else {
            return false;
        };

        // Use `TypeVisitor` instead of the output type directly to find the span of `ty` for
        // cases like `fn foo() -> (dyn Trait, i32) {}`.
        // Recursively look for `TraitObject` types and if there's only one, use that span to
        // suggest `impl Trait`.

        // Visit to make sure there's a single `return` type to suggest `impl Trait`,
        // otherwise suggest using `Box<dyn Trait>` or an enum.
        let mut visitor = ReturnsVisitor::default();
        visitor.visit_body(&body);

        let typeck_results = self.typeck_results.as_ref().unwrap();
        let Some(liberated_sig) = typeck_results.liberated_fn_sigs().get(fn_hir_id).copied() else { return false; };

        let ret_types = visitor
            .returns
            .iter()
            .filter_map(|expr| Some((expr.span, typeck_results.node_type_opt(expr.hir_id)?)))
            .map(|(expr_span, ty)| (expr_span, self.resolve_vars_if_possible(ty)));
        let (last_ty, all_returns_have_same_type, only_never_return) = ret_types.clone().fold(
            (None, true, true),
            |(last_ty, mut same, only_never_return): (std::option::Option<Ty<'_>>, bool, bool),
             (_, ty)| {
                let ty = self.resolve_vars_if_possible(ty);
                same &=
                    !matches!(ty.kind(), ty::Error(_))
                        && last_ty.map_or(true, |last_ty| {
                            // FIXME: ideally we would use `can_coerce` here instead, but `typeck` comes
                            // *after* in the dependency graph.
                            match (ty.kind(), last_ty.kind()) {
                                (Infer(InferTy::IntVar(_)), Infer(InferTy::IntVar(_)))
                                | (Infer(InferTy::FloatVar(_)), Infer(InferTy::FloatVar(_)))
                                | (Infer(InferTy::FreshIntTy(_)), Infer(InferTy::FreshIntTy(_)))
                                | (
                                    Infer(InferTy::FreshFloatTy(_)),
                                    Infer(InferTy::FreshFloatTy(_)),
                                ) => true,
                                _ => ty == last_ty,
                            }
                        });
                (Some(ty), same, only_never_return && matches!(ty.kind(), ty::Never))
            },
        );
        let mut spans_and_needs_box = vec![];

        match liberated_sig.output().kind() {
            ty::Dynamic(predicates, _, ty::Dyn) => {
                let cause = ObligationCause::misc(ret_ty.span, obligation.cause.body_id);
                let param_env = ty::ParamEnv::empty();

                if !only_never_return {
                    for (expr_span, return_ty) in ret_types {
                        let self_ty_satisfies_dyn_predicates = |self_ty| {
                            predicates.iter().all(|predicate| {
                                let pred = predicate.with_self_ty(self.tcx, self_ty);
                                let obl = Obligation::new(self.tcx, cause.clone(), param_env, pred);
                                self.predicate_may_hold(&obl)
                            })
                        };

                        if let ty::Adt(def, substs) = return_ty.kind()
                            && def.is_box()
                            && self_ty_satisfies_dyn_predicates(substs.type_at(0))
                        {
                            spans_and_needs_box.push((expr_span, false));
                        } else if self_ty_satisfies_dyn_predicates(return_ty) {
                            spans_and_needs_box.push((expr_span, true));
                        } else {
                            return false;
                        }
                    }
                }
            }
            _ => return false,
        };

        let sm = self.tcx.sess.source_map();
        if !ret_ty.span.overlaps(span) {
            return false;
        }
        let snippet = if let hir::TyKind::TraitObject(..) = ret_ty.kind {
            if let Ok(snippet) = sm.span_to_snippet(ret_ty.span) {
                snippet
            } else {
                return false;
            }
        } else {
            // Substitute the type, so we can print a fixup given `type Alias = dyn Trait`
            let name = liberated_sig.output().to_string();
            let name =
                name.strip_prefix('(').and_then(|name| name.strip_suffix(')')).unwrap_or(&name);
            if !name.starts_with("dyn ") {
                return false;
            }
            name.to_owned()
        };

        err.code(error_code!(E0746));
        err.set_primary_message("return type cannot have an unboxed trait object");
        err.children.clear();
        let impl_trait_msg = "for information on `impl Trait`, see \
            <https://doc.rust-lang.org/book/ch10-02-traits.html\
            #returning-types-that-implement-traits>";
        let trait_obj_msg = "for information on trait objects, see \
            <https://doc.rust-lang.org/book/ch17-02-trait-objects.html\
            #using-trait-objects-that-allow-for-values-of-different-types>";

        let has_dyn = snippet.split_whitespace().next().map_or(false, |s| s == "dyn");
        let trait_obj = if has_dyn { &snippet[4..] } else { &snippet };
        if only_never_return {
            // No return paths, probably using `panic!()` or similar.
            // Suggest `-> impl Trait`, and if `Trait` is object safe, `-> Box<dyn Trait>`.
            suggest_trait_object_return_type_alternatives(
                err,
                ret_ty.span,
                trait_obj,
                is_object_safe,
            );
        } else if let (Some(last_ty), true) = (last_ty, all_returns_have_same_type) {
            // Suggest `-> impl Trait`.
            err.span_suggestion(
                ret_ty.span,
                &format!(
                    "use `impl {1}` as the return type, as all return paths are of type `{}`, \
                     which implements `{1}`",
                    last_ty, trait_obj,
                ),
                format!("impl {}", trait_obj),
                Applicability::MachineApplicable,
            );
            err.note(impl_trait_msg);
        } else {
            if is_object_safe {
                // Suggest `-> Box<dyn Trait>` and `Box::new(returned_value)`.
                err.multipart_suggestion(
                    "return a boxed trait object instead",
                    vec![
                        (ret_ty.span.shrink_to_lo(), "Box<".to_string()),
                        (span.shrink_to_hi(), ">".to_string()),
                    ],
                    Applicability::MaybeIncorrect,
                );
                for (span, needs_box) in spans_and_needs_box {
                    if needs_box {
                        err.multipart_suggestion(
                            "... and box this value",
                            vec![
                                (span.shrink_to_lo(), "Box::new(".to_string()),
                                (span.shrink_to_hi(), ")".to_string()),
                            ],
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            } else {
                // This is currently not possible to trigger because E0038 takes precedence, but
                // leave it in for completeness in case anything changes in an earlier stage.
                err.note(&format!(
                    "if trait `{}` were object-safe, you could return a trait object",
                    trait_obj,
                ));
            }
            err.note(trait_obj_msg);
            err.note(&format!(
                "if all the returned values were of the same type you could use `impl {}` as the \
                 return type",
                trait_obj,
            ));
            err.note(impl_trait_msg);
            err.note("you can create a new `enum` with a variant for each returned type");
        }
        true
    }

    fn point_at_returns_when_relevant(
        &self,
        err: &mut DiagnosticBuilder<'tcx, ErrorGuaranteed>,
        obligation: &PredicateObligation<'tcx>,
    ) {
        match obligation.cause.code().peel_derives() {
            ObligationCauseCode::SizedReturnType => {}
            _ => return,
        }

        let hir = self.tcx.hir();
        let node = hir.find_by_def_id(obligation.cause.body_id);
        if let Some(hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(_, _, body_id), .. })) =
            node
        {
            let body = hir.body(*body_id);
            // Point at all the `return`s in the function as they have failed trait bounds.
            let mut visitor = ReturnsVisitor::default();
            visitor.visit_body(&body);
            let typeck_results = self.typeck_results.as_ref().unwrap();
            for expr in &visitor.returns {
                if let Some(returned_ty) = typeck_results.node_type_opt(expr.hir_id) {
                    let ty = self.resolve_vars_if_possible(returned_ty);
                    if ty.references_error() {
                        // don't print out the [type error] here
                        err.delay_as_bug();
                    } else {
                        err.span_label(
                            expr.span,
                            &format!("this returned value is of type `{}`", ty),
                        );
                    }
                }
            }
        }
    }

    fn report_closure_arg_mismatch(
        &self,
        span: Span,
        found_span: Option<Span>,
        found: ty::PolyTraitRef<'tcx>,
        expected: ty::PolyTraitRef<'tcx>,
        cause: &ObligationCauseCode<'tcx>,
        found_node: Option<Node<'_>>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        pub(crate) fn build_fn_sig_ty<'tcx>(
            infcx: &InferCtxt<'tcx>,
            trait_ref: ty::PolyTraitRef<'tcx>,
        ) -> Ty<'tcx> {
            let inputs = trait_ref.skip_binder().substs.type_at(1);
            let sig = match inputs.kind() {
                ty::Tuple(inputs) if infcx.tcx.is_fn_trait(trait_ref.def_id()) => {
                    infcx.tcx.mk_fn_sig(
                        *inputs,
                        infcx.next_ty_var(TypeVariableOrigin {
                            span: DUMMY_SP,
                            kind: TypeVariableOriginKind::MiscVariable,
                        }),
                        false,
                        hir::Unsafety::Normal,
                        abi::Abi::Rust,
                    )
                }
                _ => infcx.tcx.mk_fn_sig(
                    [inputs],
                    infcx.next_ty_var(TypeVariableOrigin {
                        span: DUMMY_SP,
                        kind: TypeVariableOriginKind::MiscVariable,
                    }),
                    false,
                    hir::Unsafety::Normal,
                    abi::Abi::Rust,
                ),
            };

            infcx.tcx.mk_fn_ptr(trait_ref.rebind(sig))
        }

        let argument_kind = match expected.skip_binder().self_ty().kind() {
            ty::Closure(..) => "closure",
            ty::Generator(..) => "generator",
            _ => "function",
        };
        let mut err = struct_span_err!(
            self.tcx.sess,
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

        self.note_conflicting_closure_bounds(cause, &mut err);

        if let Some(found_node) = found_node {
            hint_missing_borrow(self, param_env, span, found, expected, found_node, &mut err);
        }

        err
    }

    // Add a note if there are two `Fn`-family bounds that have conflicting argument
    // requirements, which will always cause a closure to have a type error.
    fn note_conflicting_closure_bounds(
        &self,
        cause: &ObligationCauseCode<'tcx>,
        err: &mut DiagnosticBuilder<'tcx, ErrorGuaranteed>,
    ) {
        // First, look for an `ExprBindingObligation`, which means we can get
        // the unsubstituted predicate list of the called function. And check
        // that the predicate that we failed to satisfy is a `Fn`-like trait.
        if let ObligationCauseCode::ExprBindingObligation(def_id, _, _, idx) = cause
            && let predicates = self.tcx.predicates_of(def_id).instantiate_identity(self.tcx)
            && let Some(pred) = predicates.predicates.get(*idx)
            && let ty::PredicateKind::Clause(ty::Clause::Trait(trait_pred)) = pred.kind().skip_binder()
            && self.tcx.is_fn_trait(trait_pred.def_id())
        {
            let expected_self =
                self.tcx.anonymize_bound_vars(pred.kind().rebind(trait_pred.self_ty()));
            let expected_substs = self
                .tcx
                .anonymize_bound_vars(pred.kind().rebind(trait_pred.trait_ref.substs));

            // Find another predicate whose self-type is equal to the expected self type,
            // but whose substs don't match.
            let other_pred = predicates.into_iter()
                .enumerate()
                .find(|(other_idx, (pred, _))| match pred.kind().skip_binder() {
                    ty::PredicateKind::Clause(ty::Clause::Trait(trait_pred))
                        if self.tcx.is_fn_trait(trait_pred.def_id())
                            && other_idx != idx
                            // Make sure that the self type matches
                            // (i.e. constraining this closure)
                            && expected_self
                                == self.tcx.anonymize_bound_vars(
                                    pred.kind().rebind(trait_pred.self_ty()),
                                )
                            // But the substs don't match (i.e. incompatible args)
                            && expected_substs
                                != self.tcx.anonymize_bound_vars(
                                    pred.kind().rebind(trait_pred.trait_ref.substs),
                                ) =>
                    {
                        true
                    }
                    _ => false,
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

    fn suggest_fully_qualified_path(
        &self,
        err: &mut Diagnostic,
        item_def_id: DefId,
        span: Span,
        trait_ref: DefId,
    ) {
        if let Some(assoc_item) = self.tcx.opt_associated_item(item_def_id) {
            if let ty::AssocKind::Const | ty::AssocKind::Type = assoc_item.kind {
                err.note(&format!(
                    "{}s cannot be accessed directly on a `trait`, they can only be \
                        accessed through a specific `impl`",
                    self.tcx.def_kind_descr(assoc_item.kind.as_def_kind(), item_def_id)
                ));
                err.span_suggestion(
                    span,
                    "use the fully qualified path to an implementation",
                    format!("<Type as {}>::{}", self.tcx.def_path_str(trait_ref), assoc_item.name),
                    Applicability::HasPlaceholders,
                );
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
    fn maybe_note_obligation_cause_for_async_await(
        &self,
        err: &mut Diagnostic,
        obligation: &PredicateObligation<'tcx>,
    ) -> bool {
        let hir = self.tcx.hir();

        // Attempt to detect an async-await error by looking at the obligation causes, looking
        // for a generator to be present.
        //
        // When a future does not implement a trait because of a captured type in one of the
        // generators somewhere in the call stack, then the result is a chain of obligations.
        //
        // Given an `async fn` A that calls an `async fn` B which captures a non-send type and that
        // future is passed as an argument to a function C which requires a `Send` type, then the
        // chain looks something like this:
        //
        // - `BuiltinDerivedObligation` with a generator witness (B)
        // - `BuiltinDerivedObligation` with a generator (B)
        // - `BuiltinDerivedObligation` with `impl std::future::Future` (B)
        // - `BuiltinDerivedObligation` with a generator witness (A)
        // - `BuiltinDerivedObligation` with a generator (A)
        // - `BuiltinDerivedObligation` with `impl std::future::Future` (A)
        // - `BindingObligation` with `impl_send` (Send requirement)
        //
        // The first obligation in the chain is the most useful and has the generator that captured
        // the type. The last generator (`outer_generator` below) has information about where the
        // bound was introduced. At least one generator should be present for this diagnostic to be
        // modified.
        let (mut trait_ref, mut target_ty) = match obligation.predicate.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::Clause::Trait(p)) => (Some(p), Some(p.self_ty())),
            _ => (None, None),
        };
        let mut generator = None;
        let mut outer_generator = None;
        let mut next_code = Some(obligation.cause.code());

        let mut seen_upvar_tys_infer_tuple = false;

        while let Some(code) = next_code {
            debug!(?code);
            match code {
                ObligationCauseCode::FunctionArgumentObligation { parent_code, .. } => {
                    next_code = Some(parent_code);
                }
                ObligationCauseCode::ImplDerivedObligation(cause) => {
                    let ty = cause.derived.parent_trait_pred.skip_binder().self_ty();
                    debug!(
                        parent_trait_ref = ?cause.derived.parent_trait_pred,
                        self_ty.kind = ?ty.kind(),
                        "ImplDerived",
                    );

                    match *ty.kind() {
                        ty::Generator(did, ..) | ty::GeneratorWitnessMIR(did, _) => {
                            generator = generator.or(Some(did));
                            outer_generator = Some(did);
                        }
                        ty::GeneratorWitness(..) => {}
                        ty::Tuple(_) if !seen_upvar_tys_infer_tuple => {
                            // By introducing a tuple of upvar types into the chain of obligations
                            // of a generator, the first non-generator item is now the tuple itself,
                            // we shall ignore this.

                            seen_upvar_tys_infer_tuple = true;
                        }
                        _ if generator.is_none() => {
                            trait_ref = Some(cause.derived.parent_trait_pred.skip_binder());
                            target_ty = Some(ty);
                        }
                        _ => {}
                    }

                    next_code = Some(&cause.derived.parent_code);
                }
                ObligationCauseCode::DerivedObligation(derived_obligation)
                | ObligationCauseCode::BuiltinDerivedObligation(derived_obligation) => {
                    let ty = derived_obligation.parent_trait_pred.skip_binder().self_ty();
                    debug!(
                        parent_trait_ref = ?derived_obligation.parent_trait_pred,
                        self_ty.kind = ?ty.kind(),
                    );

                    match *ty.kind() {
                        ty::Generator(did, ..) | ty::GeneratorWitnessMIR(did, ..) => {
                            generator = generator.or(Some(did));
                            outer_generator = Some(did);
                        }
                        ty::GeneratorWitness(..) => {}
                        ty::Tuple(_) if !seen_upvar_tys_infer_tuple => {
                            // By introducing a tuple of upvar types into the chain of obligations
                            // of a generator, the first non-generator item is now the tuple itself,
                            // we shall ignore this.

                            seen_upvar_tys_infer_tuple = true;
                        }
                        _ if generator.is_none() => {
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

        // Only continue if a generator was found.
        debug!(?generator, ?trait_ref, ?target_ty);
        let (Some(generator_did), Some(trait_ref), Some(target_ty)) = (generator, trait_ref, target_ty) else {
            return false;
        };

        let span = self.tcx.def_span(generator_did);

        let generator_did_root = self.tcx.typeck_root_def_id(generator_did);
        debug!(
            ?generator_did,
            ?generator_did_root,
            typeck_results.hir_owner = ?self.typeck_results.as_ref().map(|t| t.hir_owner),
            ?span,
        );

        let generator_body = generator_did
            .as_local()
            .and_then(|def_id| hir.maybe_body_owned_by(def_id))
            .map(|body_id| hir.body(body_id));
        let mut visitor = AwaitsVisitor::default();
        if let Some(body) = generator_body {
            visitor.visit_body(body);
        }
        debug!(awaits = ?visitor.awaits);

        // Look for a type inside the generator interior that matches the target type to get
        // a span.
        let target_ty_erased = self.tcx.erase_regions(target_ty);
        let ty_matches = |ty| -> bool {
            // Careful: the regions for types that appear in the
            // generator interior are not generally known, so we
            // want to erase them when comparing (and anyway,
            // `Send` and other bounds are generally unaffected by
            // the choice of region). When erasing regions, we
            // also have to erase late-bound regions. This is
            // because the types that appear in the generator
            // interior generally contain "bound regions" to
            // represent regions that are part of the suspended
            // generator frame. Bound regions are preserved by
            // `erase_regions` and so we must also call
            // `erase_late_bound_regions`.
            let ty_erased = self.tcx.erase_late_bound_regions(ty);
            let ty_erased = self.tcx.erase_regions(ty_erased);
            let eq = ty_erased == target_ty_erased;
            debug!(?ty_erased, ?target_ty_erased, ?eq);
            eq
        };

        // Get the typeck results from the infcx if the generator is the function we are currently
        // type-checking; otherwise, get them by performing a query. This is needed to avoid
        // cycles. If we can't use resolved types because the generator comes from another crate,
        // we still provide a targeted error but without all the relevant spans.
        let generator_data = match &self.typeck_results {
            Some(t) if t.hir_owner.to_def_id() == generator_did_root => GeneratorData::Local(&t),
            _ if generator_did.is_local() => {
                GeneratorData::Local(self.tcx.typeck(generator_did.expect_local()))
            }
            _ if let Some(generator_diag_data) = self.tcx.generator_diagnostic_data(generator_did) => {
                GeneratorData::Foreign(generator_diag_data)
            }
            _ => return false,
        };

        let generator_within_in_progress_typeck = match &self.typeck_results {
            Some(t) => t.hir_owner.to_def_id() == generator_did_root,
            _ => false,
        };

        let mut interior_or_upvar_span = None;

        let from_awaited_ty = generator_data.get_from_await_ty(self.tcx, visitor, hir, ty_matches);
        debug!(?from_awaited_ty);

        // The generator interior types share the same binders
        if let Some(cause) =
            generator_data.get_generator_interior_types().skip_binder().iter().find(
                |ty::GeneratorInteriorTypeCause { ty, .. }| {
                    ty_matches(generator_data.get_generator_interior_types().rebind(*ty))
                },
            )
        {
            let ty::GeneratorInteriorTypeCause { span, scope_span, yield_span, expr, .. } = cause;

            interior_or_upvar_span = Some(GeneratorInteriorOrUpvar::Interior(
                *span,
                Some((*scope_span, *yield_span, *expr, from_awaited_ty)),
            ));

            if interior_or_upvar_span.is_none() && generator_data.is_foreign() {
                interior_or_upvar_span = Some(GeneratorInteriorOrUpvar::Interior(*span, None));
            }
        } else if self.tcx.sess.opts.unstable_opts.drop_tracking_mir
            // Avoid disclosing internal information to downstream crates.
            && generator_did.is_local()
            // Try to avoid cycles.
            && !generator_within_in_progress_typeck
        {
            let generator_info = &self.tcx.mir_generator_witnesses(generator_did);
            debug!(?generator_info);

            'find_source: for (variant, source_info) in
                generator_info.variant_fields.iter().zip(&generator_info.variant_source_info)
            {
                debug!(?variant);
                for &local in variant {
                    let decl = &generator_info.field_tys[local];
                    debug!(?decl);
                    if ty_matches(ty::Binder::dummy(decl.ty)) && !decl.ignore_for_traits {
                        interior_or_upvar_span = Some(GeneratorInteriorOrUpvar::Interior(
                            decl.source_info.span,
                            Some((None, source_info.span, None, from_awaited_ty)),
                        ));
                        break 'find_source;
                    }
                }
            }
        }

        if interior_or_upvar_span.is_none() {
            interior_or_upvar_span =
                generator_data.try_get_upvar_span(&self, generator_did, ty_matches);
        }

        if interior_or_upvar_span.is_none() && generator_data.is_foreign() {
            interior_or_upvar_span = Some(GeneratorInteriorOrUpvar::Interior(span, None));
        }

        debug!(?interior_or_upvar_span);
        if let Some(interior_or_upvar_span) = interior_or_upvar_span {
            let is_async = self.tcx.generator_is_async(generator_did);
            let typeck_results = match generator_data {
                GeneratorData::Local(typeck_results) => Some(typeck_results),
                GeneratorData::Foreign(_) => None,
            };
            self.note_obligation_cause_for_async_await(
                err,
                interior_or_upvar_span,
                is_async,
                outer_generator,
                trait_ref,
                target_ty,
                typeck_results,
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
    fn note_obligation_cause_for_async_await(
        &self,
        err: &mut Diagnostic,
        interior_or_upvar_span: GeneratorInteriorOrUpvar,
        is_async: bool,
        outer_generator: Option<DefId>,
        trait_pred: ty::TraitPredicate<'tcx>,
        target_ty: Ty<'tcx>,
        typeck_results: Option<&ty::TypeckResults<'tcx>>,
        obligation: &PredicateObligation<'tcx>,
        next_code: Option<&ObligationCauseCode<'tcx>>,
    ) {
        let source_map = self.tcx.sess.source_map();

        let (await_or_yield, an_await_or_yield) =
            if is_async { ("await", "an await") } else { ("yield", "a yield") };
        let future_or_generator = if is_async { "future" } else { "generator" };

        // Special case the primary error message when send or sync is the trait that was
        // not implemented.
        let hir = self.tcx.hir();
        let trait_explanation = if let Some(name @ (sym::Send | sym::Sync)) =
            self.tcx.get_diagnostic_name(trait_pred.def_id())
        {
            let (trait_name, trait_verb) =
                if name == sym::Send { ("`Send`", "sent") } else { ("`Sync`", "shared") };

            err.clear_code();
            err.set_primary_message(format!(
                "{} cannot be {} between threads safely",
                future_or_generator, trait_verb
            ));

            let original_span = err.span.primary_span().unwrap();
            let mut span = MultiSpan::from_span(original_span);

            let message = outer_generator
                .and_then(|generator_did| {
                    Some(match self.tcx.generator_kind(generator_did).unwrap() {
                        GeneratorKind::Gen => format!("generator is not {}", trait_name),
                        GeneratorKind::Async(AsyncGeneratorKind::Fn) => self
                            .tcx
                            .parent(generator_did)
                            .as_local()
                            .map(|parent_did| hir.local_def_id_to_hir_id(parent_did))
                            .and_then(|parent_hir_id| hir.opt_name(parent_hir_id))
                            .map(|name| {
                                format!("future returned by `{}` is not {}", name, trait_name)
                            })?,
                        GeneratorKind::Async(AsyncGeneratorKind::Block) => {
                            format!("future created by async block is not {}", trait_name)
                        }
                        GeneratorKind::Async(AsyncGeneratorKind::Closure) => {
                            format!("future created by async closure is not {}", trait_name)
                        }
                    })
                })
                .unwrap_or_else(|| format!("{} is not {}", future_or_generator, trait_name));

            span.push_span_label(original_span, message);
            err.set_span(span);

            format!("is not {}", trait_name)
        } else {
            format!("does not implement `{}`", trait_pred.print_modifiers_and_trait_path())
        };

        let mut explain_yield =
            |interior_span: Span, yield_span: Span, scope_span: Option<Span>| {
                let mut span = MultiSpan::from_span(yield_span);
                let snippet = match source_map.span_to_snippet(interior_span) {
                    // #70935: If snippet contains newlines, display "the value" instead
                    // so that we do not emit complex diagnostics.
                    Ok(snippet) if !snippet.contains('\n') => format!("`{}`", snippet),
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
                    format!("{} occurs here, with {} maybe used later", await_or_yield, snippet),
                );
                span.push_span_label(
                    interior_span,
                    format!("has type `{}` which {}", target_ty, trait_explanation),
                );
                if let Some(scope_span) = scope_span {
                    let scope_span = source_map.end_point(scope_span);

                    let msg = format!("{} is later dropped here", snippet);
                    span.push_span_label(scope_span, msg);
                }
                err.span_note(
                    span,
                    &format!(
                        "{} {} as this value is used across {}",
                        future_or_generator, trait_explanation, an_await_or_yield
                    ),
                );
            };
        match interior_or_upvar_span {
            GeneratorInteriorOrUpvar::Interior(interior_span, interior_extra_info) => {
                if let Some((scope_span, yield_span, expr, from_awaited_ty)) = interior_extra_info {
                    if let Some(await_span) = from_awaited_ty {
                        // The type causing this obligation is one being awaited at await_span.
                        let mut span = MultiSpan::from_span(await_span);
                        span.push_span_label(
                            await_span,
                            format!(
                                "await occurs here on type `{}`, which {}",
                                target_ty, trait_explanation
                            ),
                        );
                        err.span_note(
                            span,
                            &format!(
                                "future {not_trait} as it awaits another future which {not_trait}",
                                not_trait = trait_explanation
                            ),
                        );
                    } else {
                        // Look at the last interior type to get a span for the `.await`.
                        debug!(
                            generator_interior_types = ?format_args!(
                                "{:#?}", typeck_results.as_ref().map(|t| &t.generator_interior_types)
                            ),
                        );
                        explain_yield(interior_span, yield_span, scope_span);
                    }

                    if let Some(expr_id) = expr {
                        let expr = hir.expect_expr(expr_id);
                        debug!("target_ty evaluated from {:?}", expr);

                        let parent = hir.parent_id(expr_id);
                        if let Some(hir::Node::Expr(e)) = hir.find(parent) {
                            let parent_span = hir.span(parent);
                            let parent_did = parent.owner.to_def_id();
                            // ```rust
                            // impl T {
                            //     fn foo(&self) -> i32 {}
                            // }
                            // T.foo();
                            // ^^^^^^^ a temporary `&T` created inside this method call due to `&self`
                            // ```
                            //
                            let is_region_borrow = if let Some(typeck_results) = typeck_results {
                                typeck_results
                                    .expr_adjustments(expr)
                                    .iter()
                                    .any(|adj| adj.is_region_borrow())
                            } else {
                                false
                            };

                            // ```rust
                            // struct Foo(*const u8);
                            // bar(Foo(std::ptr::null())).await;
                            //     ^^^^^^^^^^^^^^^^^^^^^ raw-ptr `*T` created inside this struct ctor.
                            // ```
                            debug!(parent_def_kind = ?self.tcx.def_kind(parent_did));
                            let is_raw_borrow_inside_fn_like_call =
                                match self.tcx.def_kind(parent_did) {
                                    DefKind::Fn | DefKind::Ctor(..) => target_ty.is_unsafe_ptr(),
                                    _ => false,
                                };
                            if let Some(typeck_results) = typeck_results {
                                if (typeck_results.is_method_call(e) && is_region_borrow)
                                    || is_raw_borrow_inside_fn_like_call
                                {
                                    err.span_help(
                                        parent_span,
                                        "consider moving this into a `let` \
                        binding to create a shorter lived borrow",
                                    );
                                }
                            }
                        }
                    }
                }
            }
            GeneratorInteriorOrUpvar::Upvar(upvar_span) => {
                // `Some((ref_ty, is_mut))` if `target_ty` is `&T` or `&mut T` and fails to impl `Send`
                let non_send = match target_ty.kind() {
                    ty::Ref(_, ref_ty, mutability) => match self.evaluate_obligation(&obligation) {
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
                                "has type `{}` which {}, because `{}` is not `{}`",
                                target_ty, trait_explanation, ref_ty, ref_ty_trait
                            ),
                            format!(
                                "captured value {} because `{}` references cannot be sent unless their referent is `{}`",
                                trait_explanation, ref_kind, ref_ty_trait
                            ),
                        )
                    }
                    None => (
                        format!("has type `{}` which {}", target_ty, trait_explanation),
                        format!("captured value {}", trait_explanation),
                    ),
                };

                let mut span = MultiSpan::from_span(upvar_span);
                span.push_span_label(upvar_span, span_label);
                err.span_note(span, &span_note);
            }
        }

        // Add a note for the item obligation that remains - normally a note pointing to the
        // bound that introduced the obligation (e.g. `T: Send`).
        debug!(?next_code);
        self.note_obligation_cause_code(
            err,
            obligation.predicate,
            obligation.param_env,
            next_code.unwrap(),
            &mut Vec::new(),
            &mut Default::default(),
        );
    }

    fn note_obligation_cause_code<T>(
        &self,
        err: &mut Diagnostic,
        predicate: T,
        param_env: ty::ParamEnv<'tcx>,
        cause_code: &ObligationCauseCode<'tcx>,
        obligated_types: &mut Vec<Ty<'tcx>>,
        seen_requirements: &mut FxHashSet<DefId>,
    ) where
        T: ToPredicate<'tcx>,
    {
        let tcx = self.tcx;
        let predicate = predicate.to_predicate(tcx);
        match *cause_code {
            ObligationCauseCode::ExprAssignable
            | ObligationCauseCode::MatchExpressionArm { .. }
            | ObligationCauseCode::Pattern { .. }
            | ObligationCauseCode::IfExpression { .. }
            | ObligationCauseCode::IfExpressionWithNoElse
            | ObligationCauseCode::MainFunctionType
            | ObligationCauseCode::StartFunctionType
            | ObligationCauseCode::IntrinsicType
            | ObligationCauseCode::MethodReceiver
            | ObligationCauseCode::ReturnNoExpression
            | ObligationCauseCode::UnifyReceiver(..)
            | ObligationCauseCode::OpaqueType
            | ObligationCauseCode::MiscObligation
            | ObligationCauseCode::WellFormed(..)
            | ObligationCauseCode::MatchImpl(..)
            | ObligationCauseCode::ReturnType
            | ObligationCauseCode::ReturnValue(_)
            | ObligationCauseCode::BlockTailExpression(_)
            | ObligationCauseCode::AwaitableExpr(_)
            | ObligationCauseCode::ForLoopIterator
            | ObligationCauseCode::QuestionMark
            | ObligationCauseCode::CheckAssociatedTypeBounds { .. }
            | ObligationCauseCode::LetElse
            | ObligationCauseCode::BinOp { .. }
            | ObligationCauseCode::AscribeUserTypeProvePredicate(..)
            | ObligationCauseCode::RustCall => {}
            ObligationCauseCode::SliceOrArrayElem => {
                err.note("slice and array elements must have `Sized` type");
            }
            ObligationCauseCode::TupleElem => {
                err.note("only the last element of a tuple may have a dynamically sized type");
            }
            ObligationCauseCode::ProjectionWf(data) => {
                err.note(&format!("required so that the projection `{data}` is well-formed"));
            }
            ObligationCauseCode::ReferenceOutlivesReferent(ref_ty) => {
                err.note(&format!(
                    "required so that reference `{ref_ty}` does not outlive its referent"
                ));
            }
            ObligationCauseCode::ObjectTypeBound(object_ty, region) => {
                err.note(&format!(
                    "required so that the lifetime bound of `{}` for `{}` is satisfied",
                    region, object_ty,
                ));
            }
            ObligationCauseCode::ItemObligation(_)
            | ObligationCauseCode::ExprItemObligation(..) => {
                // We hold the `DefId` of the item introducing the obligation, but displaying it
                // doesn't add user usable information. It always point at an associated item.
            }
            ObligationCauseCode::BindingObligation(item_def_id, span)
            | ObligationCauseCode::ExprBindingObligation(item_def_id, span, ..) => {
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
                let descr = format!("required by a bound in `{item_name}`");
                if span.is_visible(sm) {
                    let msg = format!("required by this bound in `{short_item_name}`");
                    multispan.push_span_label(span, msg);
                    err.span_note(multispan, &descr);
                } else {
                    err.span_note(tcx.def_span(item_def_id), &descr);
                }
            }
            ObligationCauseCode::ObjectCastObligation(concrete_ty, object_ty) => {
                let (concrete_ty, concrete_file) =
                    self.tcx.short_ty_string(self.resolve_vars_if_possible(concrete_ty));
                let (object_ty, object_file) =
                    self.tcx.short_ty_string(self.resolve_vars_if_possible(object_ty));
                err.note(&with_forced_trimmed_paths!(format!(
                    "required for the cast from `{concrete_ty}` to the object type `{object_ty}`",
                )));
                if let Some(file) = concrete_file {
                    err.note(&format!(
                        "the full name for the casted type has been written to '{}'",
                        file.display(),
                    ));
                }
                if let Some(file) = object_file {
                    err.note(&format!(
                        "the full name for the object type has been written to '{}'",
                        file.display(),
                    ));
                }
            }
            ObligationCauseCode::Coercion { source: _, target } => {
                err.note(&format!("required by cast to type `{}`", self.ty_to_string(target)));
            }
            ObligationCauseCode::RepeatElementCopy { is_const_fn } => {
                err.note(
                    "the `Copy` trait is required because this value will be copied for each element of the array",
                );

                if is_const_fn {
                    err.help(
                        "consider creating a new `const` item and initializing it with the result \
                        of the function call to be used in the repeat position, like \
                        `const VAL: Type = const_fn();` and `let x = [VAL; 42];`",
                    );
                }

                if self.tcx.sess.is_nightly_build() && is_const_fn {
                    err.help(
                        "create an inline `const` block, see RFC #2920 \
                         <https://github.com/rust-lang/rfcs/pull/2920> for more information",
                    );
                }
            }
            ObligationCauseCode::VariableType(hir_id) => {
                let parent_node = self.tcx.hir().parent_id(hir_id);
                match self.tcx.hir().find(parent_node) {
                    Some(Node::Local(hir::Local { ty: Some(ty), .. })) => {
                        err.span_suggestion_verbose(
                            ty.span.shrink_to_lo(),
                            "consider borrowing here",
                            "&",
                            Applicability::MachineApplicable,
                        );
                        err.note("all local variables must have a statically known size");
                    }
                    Some(Node::Local(hir::Local {
                        init: Some(hir::Expr { kind: hir::ExprKind::Index(_, _), span, .. }),
                        ..
                    })) => {
                        // When encountering an assignment of an unsized trait, like
                        // `let x = ""[..];`, provide a suggestion to borrow the initializer in
                        // order to use have a slice instead.
                        err.span_suggestion_verbose(
                            span.shrink_to_lo(),
                            "consider borrowing here",
                            "&",
                            Applicability::MachineApplicable,
                        );
                        err.note("all local variables must have a statically known size");
                    }
                    Some(Node::Param(param)) => {
                        err.span_suggestion_verbose(
                            param.ty_span.shrink_to_lo(),
                            "function arguments must have a statically known size, borrowed types \
                            always have a known size",
                            "&",
                            Applicability::MachineApplicable,
                        );
                    }
                    _ => {
                        err.note("all local variables must have a statically known size");
                    }
                }
                if !self.tcx.features().unsized_locals {
                    err.help("unsized locals are gated as an unstable feature");
                }
            }
            ObligationCauseCode::SizedArgumentType(sp) => {
                if let Some(span) = sp {
                    if let ty::PredicateKind::Clause(clause) = predicate.kind().skip_binder()
                        && let ty::Clause::Trait(trait_pred) = clause
                        && let ty::Dynamic(..) = trait_pred.self_ty().kind()
                    {
                        let span = if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span)
                            && snippet.starts_with("dyn ")
                        {
                            let pos = snippet.len() - snippet[3..].trim_start().len();
                            span.with_hi(span.lo() + BytePos(pos as u32))
                        } else {
                            span.shrink_to_lo()
                        };
                        err.span_suggestion_verbose(
                            span,
                            "you can use `impl Trait` as the argument type",
                            "impl ".to_string(),
                            Applicability::MaybeIncorrect,
                        );
                    }
                    err.span_suggestion_verbose(
                        span.shrink_to_lo(),
                        "function arguments must have a statically known size, borrowed types \
                         always have a known size",
                        "&",
                        Applicability::MachineApplicable,
                    );
                } else {
                    err.note("all function arguments must have a statically known size");
                }
                if tcx.sess.opts.unstable_features.is_nightly_build()
                    && !self.tcx.features().unsized_fn_params
                {
                    err.help("unsized fn params are gated as an unstable feature");
                }
            }
            ObligationCauseCode::SizedReturnType => {
                err.note("the return type of a function must have a statically known size");
            }
            ObligationCauseCode::SizedYieldType => {
                err.note("the yield type of a generator must have a statically known size");
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
                err.span_suggestion(
                    span.shrink_to_lo(),
                    "borrowed types always have a statically known size",
                    "&",
                    Applicability::MachineApplicable,
                );
                err.multipart_suggestion(
                    "the `Box` type always has a statically known size and allocates its contents \
                     in the heap",
                    vec![
                        (span.shrink_to_lo(), "Box<".to_string()),
                        (span.shrink_to_hi(), ">".to_string()),
                    ],
                    Applicability::MachineApplicable,
                );
            }
            ObligationCauseCode::ConstSized => {
                err.note("constant expressions must have a statically known size");
            }
            ObligationCauseCode::InlineAsmSized => {
                err.note("all inline asm arguments must have a statically known size");
            }
            ObligationCauseCode::ConstPatternStructural => {
                err.note("constants used for pattern-matching must derive `PartialEq` and `Eq`");
            }
            ObligationCauseCode::SharedStatic => {
                err.note("shared static variables must have a type that implements `Sync`");
            }
            ObligationCauseCode::BuiltinDerivedObligation(ref data) => {
                let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_pred);
                let ty = parent_trait_ref.skip_binder().self_ty();
                if parent_trait_ref.references_error() {
                    // NOTE(eddyb) this was `.cancel()`, but `err`
                    // is borrowed, so we can't fully defuse it.
                    err.downgrade_to_delayed_bug();
                    return;
                }

                // If the obligation for a tuple is set directly by a Generator or Closure,
                // then the tuple must be the one containing capture types.
                let is_upvar_tys_infer_tuple = if !matches!(ty.kind(), ty::Tuple(..)) {
                    false
                } else {
                    if let ObligationCauseCode::BuiltinDerivedObligation(data) = &*data.parent_code
                    {
                        let parent_trait_ref =
                            self.resolve_vars_if_possible(data.parent_trait_pred);
                        let nested_ty = parent_trait_ref.skip_binder().self_ty();
                        matches!(nested_ty.kind(), ty::Generator(..))
                            || matches!(nested_ty.kind(), ty::Closure(..))
                    } else {
                        false
                    }
                };

                // Don't print the tuple of capture types
                'print: {
                    if !is_upvar_tys_infer_tuple {
                        let msg = with_forced_trimmed_paths!(format!(
                            "required because it appears within the type `{ty}`",
                        ));
                        match ty.kind() {
                            ty::Adt(def, _) => match self.tcx.opt_item_ident(def.did()) {
                                Some(ident) => err.span_note(ident.span, &msg),
                                None => err.note(&msg),
                            },
                            ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }) => {
                                // If the previous type is async fn, this is the future generated by the body of an async function.
                                // Avoid printing it twice (it was already printed in the `ty::Generator` arm below).
                                let is_future = tcx.ty_is_opaque_future(ty);
                                debug!(
                                    ?obligated_types,
                                    ?is_future,
                                    "note_obligation_cause_code: check for async fn"
                                );
                                if is_future
                                    && obligated_types.last().map_or(false, |ty| match ty.kind() {
                                        ty::Generator(last_def_id, ..) => {
                                            tcx.generator_is_async(*last_def_id)
                                        }
                                        _ => false,
                                    })
                                {
                                    break 'print;
                                }
                                err.span_note(self.tcx.def_span(def_id), &msg)
                            }
                            ty::GeneratorWitness(bound_tys) => {
                                use std::fmt::Write;

                                // FIXME: this is kind of an unusual format for rustc, can we make it more clear?
                                // Maybe we should just remove this note altogether?
                                // FIXME: only print types which don't meet the trait requirement
                                let mut msg =
                                    "required because it captures the following types: ".to_owned();
                                for ty in bound_tys.skip_binder() {
                                    with_forced_trimmed_paths!(write!(msg, "`{}`, ", ty).unwrap());
                                }
                                err.note(msg.trim_end_matches(", "))
                            }
                            ty::GeneratorWitnessMIR(def_id, substs) => {
                                use std::fmt::Write;

                                // FIXME: this is kind of an unusual format for rustc, can we make it more clear?
                                // Maybe we should just remove this note altogether?
                                // FIXME: only print types which don't meet the trait requirement
                                let mut msg =
                                    "required because it captures the following types: ".to_owned();
                                for bty in tcx.generator_hidden_types(*def_id) {
                                    let ty = bty.subst(tcx, substs);
                                    write!(msg, "`{}`, ", ty).unwrap();
                                }
                                err.note(msg.trim_end_matches(", "))
                            }
                            ty::Generator(def_id, _, _) => {
                                let sp = self.tcx.def_span(def_id);

                                // Special-case this to say "async block" instead of `[static generator]`.
                                let kind = tcx.generator_kind(def_id).unwrap().descr();
                                err.span_note(
                                    sp,
                                    with_forced_trimmed_paths!(&format!(
                                        "required because it's used within this {kind}",
                                    )),
                                )
                            }
                            ty::Closure(def_id, _) => err.span_note(
                                self.tcx.def_span(def_id),
                                "required because it's used within this closure",
                            ),
                            ty::Str => err.note("`str` is considered to contain a `[u8]` slice for auto trait purposes"),
                            _ => err.note(&msg),
                        };
                    }
                }

                obligated_types.push(ty);

                let parent_predicate = parent_trait_ref;
                if !self.is_recursive_obligation(obligated_types, &data.parent_code) {
                    // #74711: avoid a stack overflow
                    ensure_sufficient_stack(|| {
                        self.note_obligation_cause_code(
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
            ObligationCauseCode::ImplDerivedObligation(ref data) => {
                let mut parent_trait_pred =
                    self.resolve_vars_if_possible(data.derived.parent_trait_pred);
                parent_trait_pred.remap_constness_diag(param_env);
                let parent_def_id = parent_trait_pred.def_id();
                let (self_ty, file) =
                    self.tcx.short_ty_string(parent_trait_pred.skip_binder().self_ty());
                let msg = format!(
                    "required for `{self_ty}` to implement `{}`",
                    parent_trait_pred.print_modifiers_and_trait_path()
                );
                let mut is_auto_trait = false;
                match self.tcx.hir().get_if_local(data.impl_or_alias_def_id) {
                    Some(Node::Item(hir::Item {
                        kind: hir::ItemKind::Trait(is_auto, ..),
                        ident,
                        ..
                    })) => {
                        // FIXME: we should do something else so that it works even on crate foreign
                        // auto traits.
                        is_auto_trait = matches!(is_auto, hir::IsAuto::Yes);
                        err.span_note(ident.span, &msg);
                    }
                    Some(Node::Item(hir::Item {
                        kind: hir::ItemKind::Impl(hir::Impl { of_trait, self_ty, .. }),
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
                        err.span_note(spans, &msg);
                    }
                    _ => {
                        err.note(&msg);
                    }
                };

                if let Some(file) = file {
                    err.note(&format!(
                        "the full type name has been written to '{}'",
                        file.display(),
                    ));
                }
                let mut parent_predicate = parent_trait_pred;
                let mut data = &data.derived;
                let mut count = 0;
                seen_requirements.insert(parent_def_id);
                if is_auto_trait {
                    // We don't want to point at the ADT saying "required because it appears within
                    // the type `X`", like we would otherwise do in test `supertrait-auto-trait.rs`.
                    while let ObligationCauseCode::BuiltinDerivedObligation(derived) =
                        &*data.parent_code
                    {
                        let child_trait_ref =
                            self.resolve_vars_if_possible(derived.parent_trait_pred);
                        let child_def_id = child_trait_ref.def_id();
                        if seen_requirements.insert(child_def_id) {
                            break;
                        }
                        data = derived;
                        parent_predicate = child_trait_ref.to_predicate(tcx);
                        parent_trait_pred = child_trait_ref;
                    }
                }
                while let ObligationCauseCode::ImplDerivedObligation(child) = &*data.parent_code {
                    // Skip redundant recursive obligation notes. See `ui/issue-20413.rs`.
                    let child_trait_pred =
                        self.resolve_vars_if_possible(child.derived.parent_trait_pred);
                    let child_def_id = child_trait_pred.def_id();
                    if seen_requirements.insert(child_def_id) {
                        break;
                    }
                    count += 1;
                    data = &child.derived;
                    parent_predicate = child_trait_pred.to_predicate(tcx);
                    parent_trait_pred = child_trait_pred;
                }
                if count > 0 {
                    err.note(&format!(
                        "{} redundant requirement{} hidden",
                        count,
                        pluralize!(count)
                    ));
                    let (self_ty, file) =
                        self.tcx.short_ty_string(parent_trait_pred.skip_binder().self_ty());
                    err.note(&format!(
                        "required for `{self_ty}` to implement `{}`",
                        parent_trait_pred.print_modifiers_and_trait_path()
                    ));
                    if let Some(file) = file {
                        err.note(&format!(
                            "the full type name has been written to '{}'",
                            file.display(),
                        ));
                    }
                }
                // #74711: avoid a stack overflow
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        err,
                        parent_predicate,
                        param_env,
                        &data.parent_code,
                        obligated_types,
                        seen_requirements,
                    )
                });
            }
            ObligationCauseCode::DerivedObligation(ref data) => {
                let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_pred);
                let parent_predicate = parent_trait_ref;
                // #74711: avoid a stack overflow
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        err,
                        parent_predicate,
                        param_env,
                        &data.parent_code,
                        obligated_types,
                        seen_requirements,
                    )
                });
            }
            ObligationCauseCode::FunctionArgumentObligation {
                arg_hir_id,
                call_hir_id,
                ref parent_code,
                ..
            } => {
                self.note_function_argument_obligation(
                    arg_hir_id,
                    err,
                    parent_code,
                    param_env,
                    predicate,
                    call_hir_id,
                );
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        err,
                        predicate,
                        param_env,
                        &parent_code,
                        obligated_types,
                        seen_requirements,
                    )
                });
            }
            ObligationCauseCode::CompareImplItemObligation { trait_item_def_id, kind, .. } => {
                let item_name = self.tcx.item_name(trait_item_def_id);
                let msg = format!(
                    "the requirement `{predicate}` appears on the `impl`'s {kind} \
                     `{item_name}` but not on the corresponding trait's {kind}",
                );
                let sp = self
                    .tcx
                    .opt_item_ident(trait_item_def_id)
                    .map(|i| i.span)
                    .unwrap_or_else(|| self.tcx.def_span(trait_item_def_id));
                let mut assoc_span: MultiSpan = sp.into();
                assoc_span.push_span_label(
                    sp,
                    format!("this trait's {kind} doesn't have the requirement `{predicate}`"),
                );
                if let Some(ident) = self
                    .tcx
                    .opt_associated_item(trait_item_def_id)
                    .and_then(|i| self.tcx.opt_item_ident(i.container_id(self.tcx)))
                {
                    assoc_span.push_span_label(ident.span, "in this trait");
                }
                err.span_note(assoc_span, &msg);
            }
            ObligationCauseCode::TrivialBound => {
                err.help("see issue #48214");
                if tcx.sess.opts.unstable_features.is_nightly_build() {
                    err.help("add `#![feature(trivial_bounds)]` to the crate attributes to enable");
                }
            }
            ObligationCauseCode::OpaqueReturnType(expr_info) => {
                if let Some((expr_ty, expr_span)) = expr_info {
                    let expr_ty = with_forced_trimmed_paths!(self.ty_to_string(expr_ty));
                    err.span_label(
                        expr_span,
                        with_forced_trimmed_paths!(format!(
                            "return type was inferred to be `{expr_ty}` here",
                        )),
                    );
                }
            }
        }
    }

    #[instrument(
        level = "debug", skip(self, err), fields(trait_pred.self_ty = ?trait_pred.self_ty())
    )]
    fn suggest_await_before_try(
        &self,
        err: &mut Diagnostic,
        obligation: &PredicateObligation<'tcx>,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        span: Span,
    ) {
        if let Some(body_id) = self.tcx.hir().maybe_body_owned_by(obligation.cause.body_id) {
            let body = self.tcx.hir().body(body_id);
            if let Some(hir::GeneratorKind::Async(_)) = body.generator_kind {
                let future_trait = self.tcx.require_lang_item(LangItem::Future, None);

                let self_ty = self.resolve_vars_if_possible(trait_pred.self_ty());
                let impls_future = self.type_implements_trait(
                    future_trait,
                    [self.tcx.erase_late_bound_regions(self_ty)],
                    obligation.param_env,
                );
                if !impls_future.must_apply_modulo_regions() {
                    return;
                }

                let item_def_id = self.tcx.associated_item_def_ids(future_trait)[0];
                // `<T as Future>::Output`
                let projection_ty = trait_pred.map_bound(|trait_pred| {
                    self.tcx.mk_projection(
                        item_def_id,
                        // Future::Output has no substs
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
                    err.span_suggestion_verbose(
                        span.with_hi(span.hi() - BytePos(1)).shrink_to_hi(),
                        "consider `await`ing on the `Future`",
                        ".await",
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
    }

    fn suggest_floating_point_literal(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_ref: &ty::PolyTraitRef<'tcx>,
    ) {
        let rhs_span = match obligation.cause.code() {
            ObligationCauseCode::BinOp { rhs_span: Some(span), is_lit, .. } if *is_lit => span,
            _ => return,
        };
        if let ty::Float(_) = trait_ref.skip_binder().self_ty().kind()
            && let ty::Infer(InferTy::IntVar(_)) = trait_ref.skip_binder().substs.type_at(1).kind()
        {
            err.span_suggestion_verbose(
                rhs_span.shrink_to_hi(),
                "consider using a floating-point literal by writing it with `.0`",
                ".0",
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn suggest_derive(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) {
        let Some(diagnostic_name) = self.tcx.get_diagnostic_name(trait_pred.def_id()) else {
            return;
        };
        let (adt, substs) = match trait_pred.skip_binder().self_ty().kind() {
            ty::Adt(adt, substs) if adt.did().is_local() => (adt, substs),
            _ => return,
        };
        let can_derive = {
            let is_derivable_trait = match diagnostic_name {
                sym::Default => !adt.is_enum(),
                sym::PartialEq | sym::PartialOrd => {
                    let rhs_ty = trait_pred.skip_binder().trait_ref.substs.type_at(1);
                    trait_pred.skip_binder().self_ty() == rhs_ty
                }
                sym::Eq | sym::Ord | sym::Clone | sym::Copy | sym::Hash | sym::Debug => true,
                _ => false,
            };
            is_derivable_trait &&
                // Ensure all fields impl the trait.
                adt.all_fields().all(|field| {
                    let field_ty = field.ty(self.tcx, substs);
                    let trait_substs = match diagnostic_name {
                        sym::PartialEq | sym::PartialOrd => {
                            Some(field_ty)
                        }
                        _ => None,
                    };
                    let trait_pred = trait_pred.map_bound_ref(|tr| ty::TraitPredicate {
                        trait_ref: self.tcx.mk_trait_ref(
                            trait_pred.def_id(),
                            [field_ty].into_iter().chain(trait_substs),
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
                &format!(
                    "consider annotating `{}` with `#[derive({})]`",
                    trait_pred.skip_binder().self_ty(),
                    diagnostic_name,
                ),
                format!("#[derive({})]\n", diagnostic_name),
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn suggest_dereferencing_index(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
    ) {
        if let ObligationCauseCode::ImplDerivedObligation(_) = obligation.cause.code()
            && self.tcx.is_diagnostic_item(sym::SliceIndex, trait_pred.skip_binder().trait_ref.def_id)
            && let ty::Slice(_) = trait_pred.skip_binder().trait_ref.substs.type_at(1).kind()
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
    fn note_function_argument_obligation(
        &self,
        arg_hir_id: HirId,
        err: &mut Diagnostic,
        parent_code: &ObligationCauseCode<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        failed_pred: ty::Predicate<'tcx>,
        call_hir_id: HirId,
    ) {
        let tcx = self.tcx;
        let hir = tcx.hir();
        if let Some(Node::Expr(expr)) = hir.find(arg_hir_id)
            && let Some(typeck_results) = &self.typeck_results
        {
            if let hir::Expr { kind: hir::ExprKind::Block(..), .. } = expr {
                let expr = expr.peel_blocks();
                let ty = typeck_results.expr_ty_adjusted_opt(expr).unwrap_or(tcx.ty_error_misc());
                let span = expr.span;
                if Some(span) != err.span.primary_span() {
                    err.span_label(
                        span,
                        if ty.references_error() {
                            String::new()
                        } else {
                            let ty = with_forced_trimmed_paths!(self.ty_to_string(ty));
                            format!("this tail expression is of type `{ty}`")
                        },
                    );
                }
            }

            // FIXME: visit the ty to see if there's any closure involved, and if there is,
            // check whether its evaluated return type is the same as the one corresponding
            // to an associated type (as seen from `trait_pred`) in the predicate. Like in
            // trait_pred `S: Sum<<Self as Iterator>::Item>` and predicate `i32: Sum<&()>`
            let mut type_diffs = vec![];

            if let ObligationCauseCode::ExprBindingObligation(def_id, _, _, idx) = parent_code.deref()
                && let Some(node_substs) = typeck_results.node_substs_opt(call_hir_id)
                && let where_clauses = self.tcx.predicates_of(def_id).instantiate(self.tcx, node_substs)
                && let Some(where_pred) = where_clauses.predicates.get(*idx)
            {
                if let Some(where_pred) = where_pred.to_opt_poly_trait_pred()
                    && let Some(failed_pred) = failed_pred.to_opt_poly_trait_pred()
                {
                    let mut c = CollectAllMismatches {
                        infcx: self.infcx,
                        param_env,
                        errors: vec![],
                    };
                    if let Ok(_) = c.relate(where_pred, failed_pred) {
                        type_diffs = c.errors;
                    }
                } else if let Some(where_pred) = where_pred.to_opt_poly_projection_pred()
                    && let Some(failed_pred) = failed_pred.to_opt_poly_projection_pred()
                    && let Some(found) = failed_pred.skip_binder().term.ty()
                {
                    type_diffs = vec![
                        Sorts(ty::error::ExpectedFound {
                            expected: self.tcx.mk_alias(ty::Projection, where_pred.skip_binder().projection_ty),
                            found,
                        }),
                    ];
                }
            }
            if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = expr.kind
                && let hir::Path { res: hir::def::Res::Local(hir_id), .. } = path
                && let Some(hir::Node::Pat(binding)) = self.tcx.hir().find(*hir_id)
                && let parent_hir_id = self.tcx.hir().parent_id(binding.hir_id)
                && let Some(hir::Node::Local(local)) = self.tcx.hir().find(parent_hir_id)
                && let Some(binding_expr) = local.init
            {
                // If the expression we're calling on is a binding, we want to point at the
                // `let` when talking about the type. Otherwise we'll point at every part
                // of the method chain with the type.
                self.point_at_chain(binding_expr, &typeck_results, type_diffs, param_env, err);
            } else {
                self.point_at_chain(expr, &typeck_results, type_diffs, param_env, err);
            }
        }
        let call_node = hir.find(call_hir_id);
        if let Some(Node::Expr(hir::Expr {
            kind: hir::ExprKind::MethodCall(path, rcvr, ..), ..
        })) = call_node
        {
            if Some(rcvr.span) == err.span.primary_span() {
                err.replace_span_with(path.ident.span, true);
            }
        }
        if let Some(Node::Expr(hir::Expr {
            kind:
                hir::ExprKind::Call(hir::Expr { span, .. }, _)
                | hir::ExprKind::MethodCall(hir::PathSegment { ident: Ident { span, .. }, .. }, ..),
            ..
        })) = hir.find(call_hir_id)
        {
            if Some(*span) != err.span.primary_span() {
                err.span_label(*span, "required by a bound introduced by this call");
            }
        }
    }

    fn point_at_chain(
        &self,
        expr: &hir::Expr<'_>,
        typeck_results: &TypeckResults<'tcx>,
        type_diffs: Vec<TypeError<'tcx>>,
        param_env: ty::ParamEnv<'tcx>,
        err: &mut Diagnostic,
    ) {
        let mut primary_spans = vec![];
        let mut span_labels = vec![];

        let tcx = self.tcx;

        let mut print_root_expr = true;
        let mut assocs = vec![];
        let mut expr = expr;
        let mut prev_ty = self.resolve_vars_if_possible(
            typeck_results.expr_ty_adjusted_opt(expr).unwrap_or(tcx.ty_error_misc()),
        );
        while let hir::ExprKind::MethodCall(_path_segment, rcvr_expr, _args, span) = expr.kind {
            // Point at every method call in the chain with the resulting type.
            // vec![1, 2, 3].iter().map(mapper).sum<i32>()
            //               ^^^^^^ ^^^^^^^^^^^
            expr = rcvr_expr;
            let assocs_in_this_method =
                self.probe_assoc_types_at_expr(&type_diffs, span, prev_ty, expr.hir_id, param_env);
            assocs.push(assocs_in_this_method);
            prev_ty = self.resolve_vars_if_possible(
                typeck_results.expr_ty_adjusted_opt(expr).unwrap_or(tcx.ty_error_misc()),
            );

            if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = expr.kind
                && let hir::Path { res: hir::def::Res::Local(hir_id), .. } = path
                && let Some(hir::Node::Pat(binding)) = self.tcx.hir().find(*hir_id)
                && let Some(parent) = self.tcx.hir().find_parent(binding.hir_id)
            {
                // We've reached the root of the method call chain...
                if let hir::Node::Local(local) = parent
                    && let Some(binding_expr) = local.init
                {
                    // ...and it is a binding. Get the binding creation and continue the chain.
                    expr = binding_expr;
                }
                if let hir::Node::Param(param) = parent {
                    // ...and it is a an fn argument.
                    let prev_ty = self.resolve_vars_if_possible(
                        typeck_results.node_type_opt(param.hir_id).unwrap_or(tcx.ty_error_misc()),
                    );
                    let assocs_in_this_method = self.probe_assoc_types_at_expr(&type_diffs, param.ty_span, prev_ty, param.hir_id, param_env);
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
        if let Some(ty) = typeck_results.expr_ty_opt(expr) && print_root_expr {
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
                    let Some((span, (assoc, ty))) = entry else { continue; };
                    if primary_spans.is_empty() || type_diffs.iter().any(|diff| {
                        let Sorts(expected_found) = diff else { return false; };
                        self.can_eq(param_env, expected_found.found, ty)
                    }) {
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
                                let Sorts(expected_found) = diff else { return false; };
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
        body_id: hir::HirId,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Vec<Option<(Span, (DefId, Ty<'tcx>))>> {
        let ocx = ObligationCtxt::new_in_snapshot(self.infcx);
        let mut assocs_in_this_method = Vec::with_capacity(type_diffs.len());
        for diff in type_diffs {
            let Sorts(expected_found) = diff else { continue; };
            let ty::Alias(ty::Projection, proj) = expected_found.expected.kind() else { continue; };

            let origin = TypeVariableOrigin { kind: TypeVariableOriginKind::TypeInference, span };
            let trait_def_id = proj.trait_def_id(self.tcx);
            // Make `Self` be equivalent to the type of the call chain
            // expression we're looking at now, so that we can tell what
            // for example `Iterator::Item` is at this point in the chain.
            let substs = InternalSubsts::for_item(self.tcx, trait_def_id, |param, _| {
                match param.kind {
                    ty::GenericParamDefKind::Type { .. } => {
                        if param.index == 0 {
                            return prev_ty.into();
                        }
                    }
                    ty::GenericParamDefKind::Lifetime | ty::GenericParamDefKind::Const { .. } => {}
                }
                self.var_for_def(span, param)
            });
            // This will hold the resolved type of the associated type, if the
            // current expression implements the trait that associated type is
            // in. For example, this would be what `Iterator::Item` is here.
            let ty_var = self.infcx.next_ty_var(origin);
            // This corresponds to `<ExprTy as Iterator>::Item = _`.
            let projection = ty::Binder::dummy(ty::PredicateKind::Clause(ty::Clause::Projection(
                ty::ProjectionPredicate {
                    projection_ty: self.tcx.mk_alias_ty(proj.def_id, substs),
                    term: ty_var.into(),
                },
            )));
            let body_def_id = self.tcx.hir().enclosing_body_owner(body_id);
            // Add `<ExprTy as Iterator>::Item = _` obligation.
            ocx.register_obligation(Obligation::misc(
                self.tcx,
                span,
                body_def_id,
                param_env,
                projection,
            ));
            if ocx.select_where_possible().is_empty() {
                // `ty_var` now holds the type that `Item` is for `ExprTy`.
                let ty_var = self.resolve_vars_if_possible(ty_var);
                assocs_in_this_method.push(Some((span, (proj.def_id, ty_var))));
            } else {
                // `<ExprTy as Iterator>` didn't select, so likely we've
                // reached the end of the iterator chain, like the originating
                // `Vec<_>`.
                // Keep the space consistent for later zipping.
                assocs_in_this_method.push(None);
            }
        }
        assocs_in_this_method
    }

    /// If the type that failed selection is an array or a reference to an array,
    /// but the trait is implemented for slices, suggest that the user converts
    /// the array into a slice.
    fn maybe_suggest_convert_to_slice(
        &self,
        err: &mut Diagnostic,
        trait_ref: ty::Binder<'tcx, ty::TraitRef<'tcx>>,
        candidate_impls: &[ImplCandidate<'tcx>],
        span: Span,
    ) {
        // Three cases where we can make a suggestion:
        // 1. `[T; _]` (array of T)
        // 2. `&[T; _]` (reference to array of T)
        // 3. `&mut [T; _]` (mutable reference to array of T)
        let (element_ty, mut mutability) = match *trait_ref.skip_binder().self_ty().kind() {
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
            ty::RawPtr(ty::TypeAndMut { ty: t, mutbl: m }) | ty::Ref(_, t, m) => {
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
            .filter(|t| is_slice(*t))
            .next()
        {
            let msg = &format!("convert the array to a `{}` slice instead", slice_ty);

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
}

/// Add a hint to add a missing borrow or remove an unnecessary one.
fn hint_missing_borrow<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
    found: Ty<'tcx>,
    expected: Ty<'tcx>,
    found_node: Node<'_>,
    err: &mut Diagnostic,
) {
    let found_args = match found.kind() {
        ty::FnPtr(f) => infcx.instantiate_binder_with_placeholders(*f).inputs().iter(),
        kind => {
            span_bug!(span, "found was converted to a FnPtr above but is now {:?}", kind)
        }
    };
    let expected_args = match expected.kind() {
        ty::FnPtr(f) => infcx.instantiate_binder_with_placeholders(*f).inputs().iter(),
        kind => {
            span_bug!(span, "expected was converted to a FnPtr above but is now {:?}", kind)
        }
    };

    // This could be a variant constructor, for example.
    let Some(fn_decl) = found_node.fn_decl() else { return; };

    let args = fn_decl.inputs.iter().map(|ty| ty);

    fn get_deref_type_and_refs(mut ty: Ty<'_>) -> (Ty<'_>, Vec<hir::Mutability>) {
        let mut refs = vec![];

        while let ty::Ref(_, new_ty, mutbl) = ty.kind() {
            ty = *new_ty;
            refs.push(*mutbl);
        }

        (ty, refs)
    }

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
                while let hir::TyKind::Ref(_, mut_ty) = &ty.kind && left > 0 {
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
        err.multipart_suggestion_verbose(
            "consider borrowing the argument",
            to_borrow,
            Applicability::MaybeIncorrect,
        );
    }

    if !remove_borrow.is_empty() {
        err.multipart_suggestion_verbose(
            "do not borrow the argument",
            remove_borrow,
            Applicability::MaybeIncorrect,
        );
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

    fn visit_body(&mut self, body: &'v hir::Body<'v>) {
        assert!(!self.in_block_tail);
        if body.generator_kind().is_none() {
            if let hir::ExprKind::Block(block, None) = body.value.kind {
                if block.expr.is_some() {
                    self.in_block_tail = true;
                }
            }
        }
        hir::intravisit::walk_body(self, body);
    }
}

/// Collect all the awaited expressions within the input expression.
#[derive(Default)]
struct AwaitsVisitor {
    awaits: Vec<hir::HirId>,
}

impl<'v> Visitor<'v> for AwaitsVisitor {
    fn visit_expr(&mut self, ex: &'v hir::Expr<'v>) {
        if let hir::ExprKind::Yield(_, hir::YieldSource::Await { expr: Some(id) }) = ex.kind {
            self.awaits.push(id)
        }
        hir::intravisit::walk_expr(self, ex)
    }
}

pub trait NextTypeParamName {
    fn next_type_param_name(&self, name: Option<&str>) -> String;
}

impl NextTypeParamName for &[hir::GenericParam<'_>] {
    fn next_type_param_name(&self, name: Option<&str>) -> String {
        // This is the list of possible parameter names that we might suggest.
        let name = name.and_then(|n| n.chars().next()).map(|c| c.to_string().to_uppercase());
        let name = name.as_deref();
        let possible_names = [name.unwrap_or("T"), "T", "U", "V", "X", "Y", "Z", "A", "B", "C"];
        let used_names = self
            .iter()
            .filter_map(|p| match p.name {
                hir::ParamName::Plain(ident) => Some(ident.name),
                _ => None,
            })
            .collect::<Vec<_>>();

        possible_names
            .iter()
            .find(|n| !used_names.contains(&Symbol::intern(n)))
            .unwrap_or(&"ParamName")
            .to_string()
    }
}

fn suggest_trait_object_return_type_alternatives(
    err: &mut Diagnostic,
    ret_ty: Span,
    trait_obj: &str,
    is_object_safe: bool,
) {
    err.span_suggestion(
        ret_ty,
        &format!(
            "use `impl {}` as the return type if all return paths have the same type but you \
                want to expose only the trait in the signature",
            trait_obj,
        ),
        format!("impl {}", trait_obj),
        Applicability::MaybeIncorrect,
    );
    if is_object_safe {
        err.multipart_suggestion(
            &format!(
                "use a boxed trait object if all return paths implement trait `{}`",
                trait_obj,
            ),
            vec![
                (ret_ty.shrink_to_lo(), "Box<".to_string()),
                (ret_ty.shrink_to_hi(), ">".to_string()),
            ],
            Applicability::MaybeIncorrect,
        );
    }
}

/// Collect the spans that we see the generic param `param_did`
struct ReplaceImplTraitVisitor<'a> {
    ty_spans: &'a mut Vec<Span>,
    param_did: DefId,
}

impl<'a, 'hir> hir::intravisit::Visitor<'hir> for ReplaceImplTraitVisitor<'a> {
    fn visit_ty(&mut self, t: &'hir hir::Ty<'hir>) {
        if let hir::TyKind::Path(hir::QPath::Resolved(
            None,
            hir::Path { res: hir::def::Res::Def(_, segment_did), .. },
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

    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}
