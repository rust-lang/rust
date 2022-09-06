use super::{
    EvaluationResult, Obligation, ObligationCause, ObligationCauseCode, PredicateObligation,
    SelectionContext,
};

use crate::autoderef::Autoderef;
use crate::infer::InferCtxt;
use crate::traits::normalize_to;

use hir::HirId;
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
use rustc_hir::lang_items::LangItem;
use rustc_hir::{AsyncGeneratorKind, GeneratorKind, Node};
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::hir::map;
use rustc_middle::ty::{
    self, suggest_arbitrary_trait_bound, suggest_constraining_type_param, AdtKind, DefIdTree,
    GeneratorDiagnosticData, GeneratorInteriorTypeCause, Infer, InferTy, IsSuggestable,
    ProjectionPredicate, ToPredicate, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeVisitable,
};
use rustc_middle::ty::{TypeAndMut, TypeckResults};
use rustc_session::Limit;
use rustc_span::def_id::LOCAL_CRATE;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{BytePos, DesugaringKind, ExpnKind, Span, DUMMY_SP};
use rustc_target::spec::abi;
use std::fmt;

use super::InferCtxtPrivExt;
use crate::infer::InferCtxtExt as _;
use crate::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_middle::ty::print::with_no_trimmed_paths;

#[derive(Debug)]
pub enum GeneratorInteriorOrUpvar {
    // span of interior type
    Interior(Span),
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
        infer_context: &InferCtxt<'a, 'tcx>,
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
                        if ty_matches(ty::Binder::dummy(upvar_ty)) {
                            Some(GeneratorInteriorOrUpvar::Upvar(upvar.span))
                        } else {
                            None
                        }
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
                                                "node_type: no type for node `{}`",
                                                ty::tls::with(|tcx| tcx
                                                    .hir()
                                                    .node_to_string(await_expr.hir_id))
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
pub trait InferCtxtExt<'tcx> {
    fn suggest_restricting_param_bound(
        &self,
        err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        proj_pred: Option<ty::PolyProjectionPredicate<'tcx>>,
        body_id: hir::HirId,
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
        err: &mut Diagnostic,
        obligation: &PredicateObligation<'tcx>,
    );

    fn report_closure_arg_mismatch(
        &self,
        span: Span,
        found_span: Option<Span>,
        found: ty::PolyTraitRef<'tcx>,
        expected: ty::PolyTraitRef<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed>;

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
        interior_extra_info: Option<(Option<Span>, Span, Option<hir::HirId>, Option<Span>)>,
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
        predicate: &T,
        param_env: ty::ParamEnv<'tcx>,
        cause_code: &ObligationCauseCode<'tcx>,
        obligated_types: &mut Vec<Ty<'tcx>>,
        seen_requirements: &mut FxHashSet<DefId>,
    ) where
        T: fmt::Display;

    fn suggest_new_overflow_limit(&self, err: &mut Diagnostic);

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
}

fn predicate_constraint(generics: &hir::Generics<'_>, pred: String) -> (Span, String) {
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
    hir_id: HirId,
    hir_generics: &hir::Generics<'tcx>,
    msg: &str,
    err: &mut Diagnostic,
    fn_sig: Option<&hir::FnSig<'_>>,
    projection: Option<&ty::ProjectionTy<'_>>,
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
    {
        return;
    }
    let Some(item_id) = hir_id.as_owner() else { return; };
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
            predicate_constraint(hir_generics, trait_pred.to_predicate(tcx).to_string()),
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
            (_, None) => {
                predicate_constraint(hir_generics, trait_pred.to_predicate(tcx).to_string())
            }
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

impl<'a, 'tcx> InferCtxtExt<'tcx> for InferCtxt<'a, 'tcx> {
    fn suggest_restricting_param_bound(
        &self,
        mut err: &mut Diagnostic,
        trait_pred: ty::PolyTraitPredicate<'tcx>,
        proj_pred: Option<ty::PolyProjectionPredicate<'tcx>>,
        body_id: hir::HirId,
    ) {
        let trait_pred = self.resolve_numeric_literals_with_default(trait_pred);

        let self_ty = trait_pred.skip_binder().self_ty();
        let (param_ty, projection) = match self_ty.kind() {
            ty::Param(_) => (true, None),
            ty::Projection(projection) => (false, Some(projection)),
            _ => (false, None),
        };

        // FIXME: Add check for trait bound that is already present, particularly `?Sized` so we
        //        don't suggest `T: Sized + ?Sized`.
        let mut hir_id = body_id;
        while let Some(node) = self.tcx.hir().find(hir_id) {
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
                        hir_id,
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
                        self.tcx, hir_id, &generics, "`Self`", err, None, projection, trait_pred,
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
                        hir_id,
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
                        hir_id,
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

                    if let Some(proj_pred) = proj_pred {
                        let ProjectionPredicate { projection_ty, term } = proj_pred.skip_binder();
                        let item = self.tcx.associated_item(projection_ty.item_def_id);

                        // FIXME: this case overlaps with code in TyCtxt::note_and_explain_type_err.
                        // That should be extracted into a helper function.
                        if constraint.ends_with('>') {
                            constraint = format!(
                                "{}, {}={}>",
                                &constraint[..constraint.len() - 1],
                                item.name,
                                term
                            );
                        } else {
                            constraint.push_str(&format!("<{}={}>", item.name, term));
                        }
                    }

                    if suggest_constraining_type_param(
                        self.tcx,
                        generics,
                        &mut err,
                        &param_name,
                        &constraint,
                        Some(trait_pred.def_id()),
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
                    if suggest_arbitrary_trait_bound(self.tcx, generics, &mut err, trait_pred) {
                        return;
                    }
                }
                hir::Node::Crate(..) => return,

                _ => {}
            }

            hir_id = self.tcx.hir().local_def_id_to_hir_id(self.tcx.hir().get_parent_item(hir_id));
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
        let ObligationCauseCode::FunctionArgumentObligation { arg_hir_id, .. } = obligation.cause.code()
            else { return false; };
        let Some(typeck_results) = self.in_progress_typeck_results
            else { return false; };
        let typeck_results = typeck_results.borrow();
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
            if self
                .can_eq(obligation.param_env, self.tcx.erase_late_bound_regions(real_ty), arg_ty)
                .is_err()
            {
                continue;
            }

            if let ty::Ref(region, base_ty, mutbl) = *real_ty.skip_binder().kind() {
                let mut autoderef = Autoderef::new(
                    self,
                    obligation.param_env,
                    obligation.cause.body_id,
                    span,
                    base_ty,
                    span,
                );
                if let Some(steps) = autoderef.find_map(|(ty, steps)| {
                    // Re-add the `&`
                    let ty = self.tcx.mk_ref(region, TypeAndMut { ty, mutbl });

                    // Remapping bound vars here
                    let real_trait_pred_and_ty =
                        real_trait_pred.map_bound(|inner_trait_pred| (inner_trait_pred, ty));
                    let obligation = self.mk_trait_obligation_with_new_self_ty(
                        obligation.param_env,
                        real_trait_pred_and_ty,
                    );
                    Some(steps).filter(|_| self.predicate_may_hold(&obligation))
                }) {
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
                        err.span_suggestion_verbose(
                            span.shrink_to_lo(),
                            "consider dereferencing here",
                            "*",
                            Applicability::MachineApplicable,
                        );
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
        let parent_node = hir.get_parent_node(hir_id);
        match hir.find(parent_node) {
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
        // Skipping binder here, remapping below
        let self_ty = trait_pred.self_ty().skip_binder();

        let (def_id, output_ty, callable) = match *self_ty.kind() {
            ty::Closure(def_id, substs) => (def_id, substs.as_closure().sig().output(), "closure"),
            ty::FnDef(def_id, _) => (def_id, self_ty.fn_sig(self.tcx).output(), "function"),
            _ => return false,
        };
        let msg = format!("use parentheses to call the {}", callable);

        // "We should really create a single list of bound vars from the combined vars
        // from the predicate and function, but instead we just liberate the function bound vars"
        let output_ty = self.tcx.liberate_late_bound_regions(def_id, output_ty);

        // Remapping bound vars here
        let trait_pred_and_self = trait_pred.map_bound(|trait_pred| (trait_pred, output_ty));

        let new_obligation =
            self.mk_trait_obligation_with_new_self_ty(obligation.param_env, trait_pred_and_self);

        match self.evaluate_obligation(&new_obligation) {
            Ok(
                EvaluationResult::EvaluatedToOk
                | EvaluationResult::EvaluatedToOkModuloRegions
                | EvaluationResult::EvaluatedToOkModuloOpaqueTypes
                | EvaluationResult::EvaluatedToAmbig,
            ) => {}
            _ => return false,
        }
        let hir = self.tcx.hir();
        // Get the name of the callable and the arguments to be used in the suggestion.
        let (snippet, sugg) = match hir.get_if_local(def_id) {
            Some(hir::Node::Expr(hir::Expr {
                kind: hir::ExprKind::Closure(hir::Closure { fn_decl, fn_decl_span, .. }),
                ..
            })) => {
                err.span_label(*fn_decl_span, "consider calling this closure");
                let Some(name) = self.get_closure_name(def_id, err, &msg) else {
                    return false;
                };
                let args = fn_decl.inputs.iter().map(|_| "_").collect::<Vec<_>>().join(", ");
                let sugg = format!("({})", args);
                (format!("{}{}", name, sugg), sugg)
            }
            Some(hir::Node::Item(hir::Item {
                ident,
                kind: hir::ItemKind::Fn(.., body_id),
                ..
            })) => {
                err.span_label(ident.span, "consider calling this function");
                let body = hir.body(*body_id);
                let args = body
                    .params
                    .iter()
                    .map(|arg| match &arg.pat.kind {
                        hir::PatKind::Binding(_, _, ident, None)
                        // FIXME: provide a better suggestion when encountering `SelfLower`, it
                        // should suggest a method call.
                        if ident.name != kw::SelfLower => ident.to_string(),
                        _ => "_".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let sugg = format!("({})", args);
                (format!("{}{}", ident, sugg), sugg)
            }
            _ => return false,
        };
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
                sugg,
                Applicability::HasPlaceholders,
            );
        } else {
            err.help(&format!("{}: `{}`", msg, snippet));
        }
        true
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
                .filter_map(|lang_item| self.tcx.lang_items().require(*lang_item).ok())
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
                        matches!(mutability, hir::Mutability::Mut),
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
                            ["&".to_string(), "&mut ".to_string()].into_iter(),
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        let is_mut = mut_ref_self_ty_satisfies_pred || ref_inner_ty_mut;
                        err.span_suggestion_verbose(
                            span.shrink_to_lo(),
                            &format!(
                                "consider{} borrowing here",
                                if is_mut { " mutably" } else { "" }
                            ),
                            format!("&{}", if is_mut { "mut " } else { "" }),
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
        let ty::Dynamic(predicates, _) = object_ty.kind() else { return; };
        let self_ref_ty = self.tcx.mk_imm_ref(self.tcx.lifetimes.re_erased, self_ty);

        for predicate in predicates.iter() {
            if !self.predicate_must_hold_modulo_regions(
                &obligation.with(predicate.with_self_ty(self.tcx, self_ref_ty)),
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
        let span = obligation.cause.span;

        let mut suggested = false;
        if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
            let refs_number =
                snippet.chars().filter(|c| !c.is_whitespace()).take_while(|c| *c == '&').count();
            if let Some('\'') = snippet.chars().filter(|c| !c.is_whitespace()).nth(refs_number) {
                // Do not suggest removal of borrow from type arguments.
                return false;
            }

            // Skipping binder here, remapping below
            let mut suggested_ty = trait_pred.self_ty().skip_binder();

            for refs_remaining in 0..refs_number {
                let ty::Ref(_, inner_ty, _) = suggested_ty.kind() else {
                    break;
                };
                suggested_ty = *inner_ty;

                // Remapping bound vars here
                let trait_pred_and_suggested_ty =
                    trait_pred.map_bound(|trait_pred| (trait_pred, suggested_ty));

                let new_obligation = self.mk_trait_obligation_with_new_self_ty(
                    obligation.param_env,
                    trait_pred_and_suggested_ty,
                );

                if self.predicate_may_hold(&new_obligation) {
                    let sp = self
                        .tcx
                        .sess
                        .source_map()
                        .span_take_while(span, |c| c.is_whitespace() || *c == '&');

                    let remove_refs = refs_remaining + 1;

                    let msg = if remove_refs == 1 {
                        "consider removing the leading `&`-reference".to_string()
                    } else {
                        format!("consider removing {} leading `&`-references", remove_refs)
                    };

                    err.span_suggestion_short(sp, &msg, "", Applicability::MachineApplicable);
                    suggested = true;
                    break;
                }
            }
        }
        suggested
    }

    fn suggest_remove_await(&self, obligation: &PredicateObligation<'tcx>, err: &mut Diagnostic) {
        let span = obligation.cause.span;

        if let ObligationCauseCode::AwaitableExpr(hir_id) = obligation.cause.code().peel_derives() {
            let hir = self.tcx.hir();
            if let Some(node) = hir_id.and_then(|hir_id| hir.find(hir_id)) {
                if let hir::Node::Expr(expr) = node {
                    // FIXME: use `obligation.predicate.kind()...trait_ref.self_ty()` to see if we have `()`
                    // and if not maybe suggest doing something else? If we kept the expression around we
                    // could also check if it is an fn call (very likely) and suggest changing *that*, if
                    // it is from the local crate.
                    err.span_suggestion_verbose(
                        expr.span.shrink_to_hi().with_hi(span.hi()),
                        "remove the `.await`",
                        "",
                        Applicability::MachineApplicable,
                    );
                    // FIXME: account for associated `async fn`s.
                    if let hir::Expr { span, kind: hir::ExprKind::Call(base, _), .. } = expr {
                        if let ty::PredicateKind::Trait(pred) =
                            obligation.predicate.kind().skip_binder()
                        {
                            err.span_label(
                                *span,
                                &format!("this call returns `{}`", pred.self_ty()),
                            );
                        }
                        if let Some(typeck_results) =
                            self.in_progress_typeck_results.map(|t| t.borrow())
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
            if trait_pred.has_infer_types_or_consts() {
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
                    if points_at_arg && mutability == hir::Mutability::Not && refs_number > 0 {
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
        let parent_node = hir.get_parent_node(obligation.cause.body_id);
        let node = hir.find(parent_node);
        if let Some(hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(sig, _, body_id), .. })) = node
            && let hir::ExprKind::Block(blk, _) = &hir.body(*body_id).value.kind
            && sig.decl.output.span().overlaps(span)
            && blk.expr.is_none()
            && trait_pred.self_ty().skip_binder().is_unit()
            && let Some(stmt) = blk.stmts.last()
            && let hir::StmtKind::Semi(expr) = stmt.kind
            // Only suggest this if the expression behind the semicolon implements the predicate
            && let Some(typeck_results) = self.in_progress_typeck_results
            && let Some(ty) = typeck_results.borrow().expr_ty_opt(expr)
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
        let parent_node = hir.get_parent_node(obligation.cause.body_id);
        let Some(hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(sig, ..), .. })) = hir.find(parent_node) else {
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
        let fn_hir_id = hir.get_parent_node(obligation.cause.body_id);
        let node = hir.find(fn_hir_id);
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
            ty::Dynamic(predicates, _) => {
                // If the `dyn Trait` is not object safe, do not suggest `Box<dyn Trait>`.
                predicates
                    .principal_def_id()
                    .map_or(true, |def_id| self.tcx.object_safety_violations(def_id).is_empty())
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

        let typeck_results = self.in_progress_typeck_results.map(|t| t.borrow()).unwrap();
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
            ty::Dynamic(predicates, _) => {
                let cause = ObligationCause::misc(ret_ty.span, fn_hir_id);
                let param_env = ty::ParamEnv::empty();

                if !only_never_return {
                    for (expr_span, return_ty) in ret_types {
                        let self_ty_satisfies_dyn_predicates = |self_ty| {
                            predicates.iter().all(|predicate| {
                                let pred = predicate.with_self_ty(self.tcx, self_ty);
                                let obl = Obligation::new(cause.clone(), param_env, pred);
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
            // Suggest `-> T`, `-> impl Trait`, and if `Trait` is object safe, `-> Box<dyn Trait>`.
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
        err: &mut Diagnostic,
        obligation: &PredicateObligation<'tcx>,
    ) {
        match obligation.cause.code().peel_derives() {
            ObligationCauseCode::SizedReturnType => {}
            _ => return,
        }

        let hir = self.tcx.hir();
        let parent_node = hir.get_parent_node(obligation.cause.body_id);
        let node = hir.find(parent_node);
        if let Some(hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(_, _, body_id), .. })) =
            node
        {
            let body = hir.body(*body_id);
            // Point at all the `return`s in the function as they have failed trait bounds.
            let mut visitor = ReturnsVisitor::default();
            visitor.visit_body(&body);
            let typeck_results = self.in_progress_typeck_results.map(|t| t.borrow()).unwrap();
            for expr in &visitor.returns {
                if let Some(returned_ty) = typeck_results.node_type_opt(expr.hir_id) {
                    let ty = self.resolve_vars_if_possible(returned_ty);
                    err.span_label(expr.span, &format!("this returned value is of type `{}`", ty));
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
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        pub(crate) fn build_fn_sig_ty<'tcx>(
            infcx: &InferCtxt<'_, 'tcx>,
            trait_ref: ty::PolyTraitRef<'tcx>,
        ) -> Ty<'tcx> {
            let inputs = trait_ref.skip_binder().substs.type_at(1);
            let sig = match inputs.kind() {
                ty::Tuple(inputs)
                    if infcx.tcx.fn_trait_kind_from_lang_item(trait_ref.def_id()).is_some() =>
                {
                    infcx.tcx.mk_fn_sig(
                        inputs.iter(),
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
                    std::iter::once(inputs),
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

        err
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
                    assoc_item.kind.as_def_kind().descr(item_def_id)
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
        // - `BuiltinDerivedObligation` with `std::future::GenFuture` (B)
        // - `BuiltinDerivedObligation` with `impl std::future::Future` (B)
        // - `BuiltinDerivedObligation` with `impl std::future::Future` (B)
        // - `BuiltinDerivedObligation` with a generator witness (A)
        // - `BuiltinDerivedObligation` with a generator (A)
        // - `BuiltinDerivedObligation` with `std::future::GenFuture` (A)
        // - `BuiltinDerivedObligation` with `impl std::future::Future` (A)
        // - `BuiltinDerivedObligation` with `impl std::future::Future` (A)
        // - `BindingObligation` with `impl_send (Send requirement)
        //
        // The first obligation in the chain is the most useful and has the generator that captured
        // the type. The last generator (`outer_generator` below) has information about where the
        // bound was introduced. At least one generator should be present for this diagnostic to be
        // modified.
        let (mut trait_ref, mut target_ty) = match obligation.predicate.kind().skip_binder() {
            ty::PredicateKind::Trait(p) => (Some(p), Some(p.self_ty())),
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
                        ty::Generator(did, ..) => {
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
                        ty::Generator(did, ..) => {
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

        let in_progress_typeck_results = self.in_progress_typeck_results.map(|t| t.borrow());
        let generator_did_root = self.tcx.typeck_root_def_id(generator_did);
        debug!(
            ?generator_did,
            ?generator_did_root,
            in_progress_typeck_results.hir_owner = ?in_progress_typeck_results.as_ref().map(|t| t.hir_owner),
            ?span,
        );

        let generator_body = generator_did
            .as_local()
            .and_then(|def_id| hir.maybe_body_owned_by(def_id))
            .map(|body_id| hir.body(body_id));
        let is_async = match generator_did.as_local() {
            Some(_) => generator_body
                .and_then(|body| body.generator_kind())
                .map(|generator_kind| matches!(generator_kind, hir::GeneratorKind::Async(..)))
                .unwrap_or(false),
            None => self
                .tcx
                .generator_kind(generator_did)
                .map(|generator_kind| matches!(generator_kind, hir::GeneratorKind::Async(..)))
                .unwrap_or(false),
        };
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
            // the choice of region).  When erasing regions, we
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

        let mut interior_or_upvar_span = None;
        let mut interior_extra_info = None;

        // Get the typeck results from the infcx if the generator is the function we are currently
        // type-checking; otherwise, get them by performing a query.  This is needed to avoid
        // cycles. If we can't use resolved types because the generator comes from another crate,
        // we still provide a targeted error but without all the relevant spans.
        let generator_data: Option<GeneratorData<'tcx, '_>> = match &in_progress_typeck_results {
            Some(t) if t.hir_owner.to_def_id() == generator_did_root => {
                Some(GeneratorData::Local(&t))
            }
            _ if generator_did.is_local() => {
                Some(GeneratorData::Local(self.tcx.typeck(generator_did.expect_local())))
            }
            _ => self
                .tcx
                .generator_diagnostic_data(generator_did)
                .as_ref()
                .map(|generator_diag_data| GeneratorData::Foreign(generator_diag_data)),
        };

        if let Some(generator_data) = generator_data.as_ref() {
            interior_or_upvar_span =
                generator_data.try_get_upvar_span(&self, generator_did, ty_matches);

            // The generator interior types share the same binders
            if let Some(cause) =
                generator_data.get_generator_interior_types().skip_binder().iter().find(
                    |ty::GeneratorInteriorTypeCause { ty, .. }| {
                        ty_matches(generator_data.get_generator_interior_types().rebind(*ty))
                    },
                )
            {
                let from_awaited_ty = generator_data.get_from_await_ty(visitor, hir, ty_matches);
                let ty::GeneratorInteriorTypeCause { span, scope_span, yield_span, expr, .. } =
                    cause;

                interior_or_upvar_span = Some(GeneratorInteriorOrUpvar::Interior(*span));
                interior_extra_info = Some((*scope_span, *yield_span, *expr, from_awaited_ty));
            }

            if interior_or_upvar_span.is_none() && generator_data.is_foreign() {
                interior_or_upvar_span = Some(GeneratorInteriorOrUpvar::Interior(span));
            }
        }

        if let Some(interior_or_upvar_span) = interior_or_upvar_span {
            let typeck_results = generator_data.and_then(|generator_data| match generator_data {
                GeneratorData::Local(typeck_results) => Some(typeck_results),
                GeneratorData::Foreign(_) => None,
            });
            self.note_obligation_cause_for_async_await(
                err,
                interior_or_upvar_span,
                interior_extra_info,
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
        interior_extra_info: Option<(Option<Span>, Span, Option<hir::HirId>, Option<Span>)>,
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

        let mut explain_yield = |interior_span: Span,
                                 yield_span: Span,
                                 scope_span: Option<Span>| {
            let mut span = MultiSpan::from_span(yield_span);
            if let Ok(snippet) = source_map.span_to_snippet(interior_span) {
                // #70935: If snippet contains newlines, display "the value" instead
                // so that we do not emit complex diagnostics.
                let snippet = &format!("`{}`", snippet);
                let snippet = if snippet.contains('\n') { "the value" } else { snippet };
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
                // If available, use the scope span to annotate the drop location.
                let mut scope_note = None;
                if let Some(scope_span) = scope_span {
                    let scope_span = source_map.end_point(scope_span);

                    let msg = format!("{} is later dropped here", snippet);
                    if source_map.is_multiline(yield_span.between(scope_span)) {
                        span.push_span_label(scope_span, msg);
                    } else {
                        scope_note = Some((scope_span, msg));
                    }
                }
                err.span_note(
                    span,
                    &format!(
                        "{} {} as this value is used across {}",
                        future_or_generator, trait_explanation, an_await_or_yield
                    ),
                );
                if let Some((span, msg)) = scope_note {
                    err.span_note(span, &msg);
                }
            }
        };
        match interior_or_upvar_span {
            GeneratorInteriorOrUpvar::Interior(interior_span) => {
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

                        let parent = hir.get_parent_node(expr_id);
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
                // `Some(ref_ty)` if `target_ty` is `&T` and `T` fails to impl `Sync`
                let refers_to_non_sync = match target_ty.kind() {
                    ty::Ref(_, ref_ty, _) => match self.evaluate_obligation(&obligation) {
                        Ok(eval) if !eval.may_apply() => Some(ref_ty),
                        _ => None,
                    },
                    _ => None,
                };

                let (span_label, span_note) = match refers_to_non_sync {
                    // if `target_ty` is `&T` and `T` fails to impl `Sync`,
                    // include suggestions to make `T: Sync` so that `&T: Send`
                    Some(ref_ty) => (
                        format!(
                            "has type `{}` which {}, because `{}` is not `Sync`",
                            target_ty, trait_explanation, ref_ty
                        ),
                        format!(
                            "captured value {} because `&` references cannot be sent unless their referent is `Sync`",
                            trait_explanation
                        ),
                    ),
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
            &obligation.predicate,
            obligation.param_env,
            next_code.unwrap(),
            &mut Vec::new(),
            &mut Default::default(),
        );
    }

    fn note_obligation_cause_code<T>(
        &self,
        err: &mut Diagnostic,
        predicate: &T,
        param_env: ty::ParamEnv<'tcx>,
        cause_code: &ObligationCauseCode<'tcx>,
        obligated_types: &mut Vec<Ty<'tcx>>,
        seen_requirements: &mut FxHashSet<DefId>,
    ) where
        T: fmt::Display,
    {
        let tcx = self.tcx;
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
            | ObligationCauseCode::BinOp { .. } => {}
            ObligationCauseCode::SliceOrArrayElem => {
                err.note("slice and array elements must have `Sized` type");
            }
            ObligationCauseCode::TupleElem => {
                err.note("only the last element of a tuple may have a dynamically sized type");
            }
            ObligationCauseCode::ProjectionWf(data) => {
                err.note(&format!("required so that the projection `{}` is well-formed", data,));
            }
            ObligationCauseCode::ReferenceOutlivesReferent(ref_ty) => {
                err.note(&format!(
                    "required so that reference `{}` does not outlive its referent",
                    ref_ty,
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
                let mut multispan = MultiSpan::from(span);
                if let Some(ident) = tcx.opt_item_ident(item_def_id) {
                    let sm = tcx.sess.source_map();
                    let same_line =
                        match (sm.lookup_line(ident.span.hi()), sm.lookup_line(span.lo())) {
                            (Ok(l), Ok(r)) => l.line == r.line,
                            _ => true,
                        };
                    if !ident.span.overlaps(span) && !same_line {
                        multispan.push_span_label(ident.span, "required by a bound in this");
                    }
                }
                let descr = format!("required by a bound in `{}`", item_name);
                if span != DUMMY_SP {
                    let msg = format!("required by this bound in `{}`", item_name);
                    multispan.push_span_label(span, msg);
                    err.span_note(multispan, &descr);
                } else {
                    err.span_note(tcx.def_span(item_def_id), &descr);
                }
            }
            ObligationCauseCode::ObjectCastObligation(concrete_ty, object_ty) => {
                err.note(&format!(
                    "required for the cast from `{}` to the object type `{}`",
                    self.ty_to_string(concrete_ty),
                    self.ty_to_string(object_ty)
                ));
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
                let parent_node = self.tcx.hir().get_parent_node(hir_id);
                match self.tcx.hir().find(parent_node) {
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
            ObligationCauseCode::SizedBoxType => {
                err.note("the type of a box expression must have a statically known size");
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

                let from_generator = tcx.lang_items().from_generator_fn().unwrap();

                // Don't print the tuple of capture types
                'print: {
                    if !is_upvar_tys_infer_tuple {
                        let msg = format!("required because it appears within the type `{}`", ty);
                        match ty.kind() {
                            ty::Adt(def, _) => {
                                // `gen_future` is used in all async functions; it doesn't add any additional info.
                                if self.tcx.is_diagnostic_item(sym::gen_future, def.did()) {
                                    break 'print;
                                }
                                match self.tcx.opt_item_ident(def.did()) {
                                    Some(ident) => err.span_note(ident.span, &msg),
                                    None => err.note(&msg),
                                }
                            }
                            ty::Opaque(def_id, _) => {
                                // Avoid printing the future from `core::future::from_generator`, it's not helpful
                                if tcx.parent(*def_id) == from_generator {
                                    break 'print;
                                }

                                // If the previous type is `from_generator`, this is the future generated by the body of an async function.
                                // Avoid printing it twice (it was already printed in the `ty::Generator` arm below).
                                let is_future = tcx.ty_is_opaque_future(ty);
                                debug!(
                                    ?obligated_types,
                                    ?is_future,
                                    "note_obligation_cause_code: check for async fn"
                                );
                                if is_future
                                    && obligated_types.last().map_or(false, |ty| match ty.kind() {
                                        ty::Opaque(last_def_id, _) => {
                                            tcx.parent(*last_def_id) == from_generator
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
                                    write!(msg, "`{}`, ", ty).unwrap();
                                }
                                err.note(msg.trim_end_matches(", "))
                            }
                            ty::Generator(def_id, _, _) => {
                                let sp = self.tcx.def_span(def_id);

                                // Special-case this to say "async block" instead of `[static generator]`.
                                let kind = tcx.generator_kind(def_id).unwrap();
                                err.span_note(
                                    sp,
                                    &format!("required because it's used within this {}", kind),
                                )
                            }
                            ty::Closure(def_id, _) => err.span_note(
                                self.tcx.def_span(def_id),
                                &format!("required because it's used within this closure"),
                            ),
                            _ => err.note(&msg),
                        };
                    }
                }

                obligated_types.push(ty);

                let parent_predicate = parent_trait_ref.to_predicate(tcx);
                if !self.is_recursive_obligation(obligated_types, &data.parent_code) {
                    // #74711: avoid a stack overflow
                    ensure_sufficient_stack(|| {
                        self.note_obligation_cause_code(
                            err,
                            &parent_predicate,
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
                            &parent_predicate,
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
                let msg = format!(
                    "required for `{}` to implement `{}`",
                    parent_trait_pred.skip_binder().self_ty(),
                    parent_trait_pred.print_modifiers_and_trait_path()
                );
                let mut is_auto_trait = false;
                match self.tcx.hir().get_if_local(data.impl_def_id) {
                    Some(Node::Item(hir::Item {
                        kind: hir::ItemKind::Trait(is_auto, ..),
                        ident,
                        ..
                    })) => {
                        // FIXME: we should do something else so that it works even on crate foreign
                        // auto traits.
                        is_auto_trait = matches!(is_auto, hir::IsAuto::Yes);
                        err.span_note(ident.span, &msg)
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
                        err.span_note(spans, &msg)
                    }
                    _ => err.note(&msg),
                };

                let mut parent_predicate = parent_trait_pred.to_predicate(tcx);
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
                    err.note(&format!(
                        "required for `{}` to implement `{}`",
                        parent_trait_pred.skip_binder().self_ty(),
                        parent_trait_pred.print_modifiers_and_trait_path()
                    ));
                }
                // #74711: avoid a stack overflow
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        err,
                        &parent_predicate,
                        param_env,
                        &data.parent_code,
                        obligated_types,
                        seen_requirements,
                    )
                });
            }
            ObligationCauseCode::DerivedObligation(ref data) => {
                let parent_trait_ref = self.resolve_vars_if_possible(data.parent_trait_pred);
                let parent_predicate = parent_trait_ref.to_predicate(tcx);
                // #74711: avoid a stack overflow
                ensure_sufficient_stack(|| {
                    self.note_obligation_cause_code(
                        err,
                        &parent_predicate,
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
            } => {
                let hir = self.tcx.hir();
                if let Some(Node::Expr(expr @ hir::Expr { kind: hir::ExprKind::Block(..), .. })) =
                    hir.find(arg_hir_id)
                {
                    let in_progress_typeck_results =
                        self.in_progress_typeck_results.map(|t| t.borrow());
                    let parent_id = hir.get_parent_item(arg_hir_id);
                    let typeck_results: &TypeckResults<'tcx> = match &in_progress_typeck_results {
                        Some(t) if t.hir_owner == parent_id => t,
                        _ => self.tcx.typeck(parent_id),
                    };
                    let ty = typeck_results.expr_ty_adjusted(expr);
                    let span = expr.peel_blocks().span;
                    if Some(span) != err.span.primary_span() {
                        err.span_label(
                            span,
                            &if ty.references_error() {
                                String::new()
                            } else {
                                format!("this tail expression is of type `{:?}`", ty)
                            },
                        );
                    }
                }
                if let Some(Node::Expr(hir::Expr {
                    kind:
                        hir::ExprKind::Call(hir::Expr { span, .. }, _)
                        | hir::ExprKind::MethodCall(
                            hir::PathSegment { ident: Ident { span, .. }, .. },
                            ..,
                        ),
                    ..
                })) = hir.find(call_hir_id)
                {
                    if Some(*span) != err.span.primary_span() {
                        err.span_label(*span, "required by a bound introduced by this call");
                    }
                }
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
                    "the requirement `{}` appears on the `impl`'s {kind} `{}` but not on the \
                     corresponding trait's {kind}",
                    predicate, item_name,
                );
                let sp = self
                    .tcx
                    .opt_item_ident(trait_item_def_id)
                    .map(|i| i.span)
                    .unwrap_or_else(|| self.tcx.def_span(trait_item_def_id));
                let mut assoc_span: MultiSpan = sp.into();
                assoc_span.push_span_label(
                    sp,
                    format!("this trait's {kind} doesn't have the requirement `{}`", predicate),
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
                    let expr_ty = self.resolve_vars_if_possible(expr_ty);
                    err.span_label(
                        expr_span,
                        format!("return type was inferred to be `{expr_ty}` here"),
                    );
                }
            }
        }
    }

    fn suggest_new_overflow_limit(&self, err: &mut Diagnostic) {
        let suggested_limit = match self.tcx.recursion_limit() {
            Limit(0) => Limit(2),
            limit => limit * 2,
        };
        err.help(&format!(
            "consider increasing the recursion limit by adding a \
             `#![recursion_limit = \"{}\"]` attribute to your crate (`{}`)",
            suggested_limit,
            self.tcx.crate_name(LOCAL_CRATE),
        ));
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
        let body_hir_id = obligation.cause.body_id;
        let item_id = self.tcx.hir().get_parent_node(body_hir_id);

        if let Some(body_id) =
            self.tcx.hir().maybe_body_owned_by(self.tcx.hir().local_def_id(item_id))
        {
            let body = self.tcx.hir().body(body_id);
            if let Some(hir::GeneratorKind::Async(_)) = body.generator_kind {
                let future_trait = self.tcx.require_lang_item(LangItem::Future, None);

                let self_ty = self.resolve_vars_if_possible(trait_pred.self_ty());
                let impls_future = self.type_implements_trait(
                    future_trait,
                    self.tcx.erase_late_bound_regions(self_ty),
                    ty::List::empty(),
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
                        self.tcx.mk_substs_trait(trait_pred.self_ty(), &[]),
                    )
                });
                let projection_ty = normalize_to(
                    &mut SelectionContext::new(self),
                    obligation.param_env,
                    obligation.cause.clone(),
                    projection_ty,
                    &mut vec![],
                );

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
        match (
            trait_ref.skip_binder().self_ty().kind(),
            trait_ref.skip_binder().substs.type_at(1).kind(),
        ) {
            (ty::Float(_), ty::Infer(InferTy::IntVar(_))) => {
                err.span_suggestion_verbose(
                    rhs_span.shrink_to_hi(),
                    "consider using a floating-point literal by writing it with `.0`",
                    ".0",
                    Applicability::MaybeIncorrect,
                );
            }
            _ => {}
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
                            self.tcx.mk_substs_trait(field_ty, &[field_ty.into()])
                        }
                        _ => self.tcx.mk_substs_trait(field_ty, &[]),
                    };
                    let trait_pred = trait_pred.map_bound_ref(|tr| ty::TraitPredicate {
                        trait_ref: ty::TraitRef {
                            substs: trait_substs,
                            ..trait_pred.skip_binder().trait_ref
                        },
                        ..*tr
                    });
                    let field_obl = Obligation::new(
                        obligation.cause.clone(),
                        obligation.param_env,
                        trait_pred.to_predicate(self.tcx),
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
        "use some type `T` that is `T: Sized` as the return type if all return paths have the \
            same type",
        "T",
        Applicability::MaybeIncorrect,
    );
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

impl<'tcx> TypeFolder<'tcx> for ReplaceImplTraitFolder<'tcx> {
    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Param(ty::ParamTy { index, .. }) = t.kind() {
            if self.param.index == *index {
                return self.replace_ty;
            }
        }
        t.super_fold_with(self)
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}
