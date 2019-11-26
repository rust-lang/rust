use super::{
    ConstEvalFailure,
    EvaluationResult,
    FulfillmentError,
    FulfillmentErrorCode,
    MismatchedProjectionTypes,
    ObjectSafetyViolation,
    Obligation,
    ObligationCause,
    ObligationCauseCode,
    OnUnimplementedDirective,
    OnUnimplementedNote,
    OutputTypeParameterMismatch,
    Overflow,
    PredicateObligation,
    SelectionContext,
    SelectionError,
    TraitNotObjectSafe,
};

use crate::hir;
use crate::hir::Node;
use crate::hir::def_id::DefId;
use crate::infer::{self, InferCtxt};
use crate::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use crate::session::DiagnosticMessageId;
use crate::ty::{self, AdtKind, DefIdTree, ToPredicate, ToPolyTraitRef, Ty, TyCtxt, TypeFoldable};
use crate::ty::GenericParamDefKind;
use crate::ty::error::ExpectedFound;
use crate::ty::fast_reject;
use crate::ty::fold::TypeFolder;
use crate::ty::subst::Subst;
use crate::ty::SubtypePredicate;
use crate::util::nodemap::{FxHashMap, FxHashSet};

use errors::{Applicability, DiagnosticBuilder, pluralize, Style};
use std::fmt;
use syntax::ast;
use syntax::symbol::{sym, kw};
use syntax_pos::{DUMMY_SP, Span, ExpnKind, MultiSpan};
use rustc::hir::def_id::LOCAL_CRATE;

use rustc_error_codes::*;

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub fn report_fulfillment_errors(
        &self,
        errors: &[FulfillmentError<'tcx>],
        body_id: Option<hir::BodyId>,
        fallback_has_occurred: bool,
    ) {
        #[derive(Debug)]
        struct ErrorDescriptor<'tcx> {
            predicate: ty::Predicate<'tcx>,
            index: Option<usize>, // None if this is an old error
        }

        let mut error_map: FxHashMap<_, Vec<_>> =
            self.reported_trait_errors.borrow().iter().map(|(&span, predicates)| {
                (span, predicates.iter().map(|predicate| ErrorDescriptor {
                    predicate: predicate.clone(),
                    index: None
                }).collect())
            }).collect();

        for (index, error) in errors.iter().enumerate() {
            // We want to ignore desugarings here: spans are equivalent even
            // if one is the result of a desugaring and the other is not.
            let mut span = error.obligation.cause.span;
            let expn_data = span.ctxt().outer_expn_data();
            if let ExpnKind::Desugaring(_) = expn_data.kind {
                span = expn_data.call_site;
            }

            error_map.entry(span).or_default().push(
                ErrorDescriptor {
                    predicate: error.obligation.predicate.clone(),
                    index: Some(index)
                }
            );

            self.reported_trait_errors.borrow_mut()
                .entry(span).or_default()
                .push(error.obligation.predicate.clone());
        }

        // We do this in 2 passes because we want to display errors in order, though
        // maybe it *is* better to sort errors by span or something.
        let mut is_suppressed = vec![false; errors.len()];
        for (_, error_set) in error_map.iter() {
            // We want to suppress "duplicate" errors with the same span.
            for error in error_set {
                if let Some(index) = error.index {
                    // Suppress errors that are either:
                    // 1) strictly implied by another error.
                    // 2) implied by an error with a smaller index.
                    for error2 in error_set {
                        if error2.index.map_or(false, |index2| is_suppressed[index2]) {
                            // Avoid errors being suppressed by already-suppressed
                            // errors, to prevent all errors from being suppressed
                            // at once.
                            continue
                        }

                        if self.error_implies(&error2.predicate, &error.predicate) &&
                            !(error2.index >= error.index &&
                              self.error_implies(&error.predicate, &error2.predicate))
                        {
                            info!("skipping {:?} (implied by {:?})", error, error2);
                            is_suppressed[index] = true;
                            break
                        }
                    }
                }
            }
        }

        for (error, suppressed) in errors.iter().zip(is_suppressed) {
            if !suppressed {
                self.report_fulfillment_error(error, body_id, fallback_has_occurred);
            }
        }
    }

    // returns if `cond` not occurring implies that `error` does not occur - i.e., that
    // `error` occurring implies that `cond` occurs.
    fn error_implies(
        &self,
        cond: &ty::Predicate<'tcx>,
        error: &ty::Predicate<'tcx>,
    ) -> bool {
        if cond == error {
            return true
        }

        let (cond, error) = match (cond, error) {
            (&ty::Predicate::Trait(..), &ty::Predicate::Trait(ref error))
                => (cond, error),
            _ => {
                // FIXME: make this work in other cases too.
                return false
            }
        };

        for implication in super::elaborate_predicates(self.tcx, vec![cond.clone()]) {
            if let ty::Predicate::Trait(implication) = implication {
                let error = error.to_poly_trait_ref();
                let implication = implication.to_poly_trait_ref();
                // FIXME: I'm just not taking associated types at all here.
                // Eventually I'll need to implement param-env-aware
                // `Γ₁ ⊦ φ₁ => Γ₂ ⊦ φ₂` logic.
                let param_env = ty::ParamEnv::empty();
                if self.can_sub(param_env, error, implication).is_ok() {
                    debug!("error_implies: {:?} -> {:?} -> {:?}", cond, error, implication);
                    return true
                }
            }
        }

        false
    }

    fn report_fulfillment_error(
        &self,
        error: &FulfillmentError<'tcx>,
        body_id: Option<hir::BodyId>,
        fallback_has_occurred: bool,
    ) {
        debug!("report_fulfillment_errors({:?})", error);
        match error.code {
            FulfillmentErrorCode::CodeSelectionError(ref selection_error) => {
                self.report_selection_error(
                    &error.obligation,
                    selection_error,
                    fallback_has_occurred,
                    error.points_at_arg_span,
                );
            }
            FulfillmentErrorCode::CodeProjectionError(ref e) => {
                self.report_projection_error(&error.obligation, e);
            }
            FulfillmentErrorCode::CodeAmbiguity => {
                self.maybe_report_ambiguity(&error.obligation, body_id);
            }
            FulfillmentErrorCode::CodeSubtypeError(ref expected_found, ref err) => {
                self.report_mismatched_types(
                    &error.obligation.cause,
                    expected_found.expected,
                    expected_found.found,
                    err.clone(),
                ).emit();
            }
        }
    }

    fn report_projection_error(
        &self,
        obligation: &PredicateObligation<'tcx>,
        error: &MismatchedProjectionTypes<'tcx>,
    ) {
        let predicate = self.resolve_vars_if_possible(&obligation.predicate);

        if predicate.references_error() {
            return
        }

        self.probe(|_| {
            let err_buf;
            let mut err = &error.err;
            let mut values = None;

            // try to find the mismatched types to report the error with.
            //
            // this can fail if the problem was higher-ranked, in which
            // cause I have no idea for a good error message.
            if let ty::Predicate::Projection(ref data) = predicate {
                let mut selcx = SelectionContext::new(self);
                let (data, _) = self.replace_bound_vars_with_fresh_vars(
                    obligation.cause.span,
                    infer::LateBoundRegionConversionTime::HigherRankedType,
                    data
                );
                let mut obligations = vec![];
                let normalized_ty = super::normalize_projection_type(
                    &mut selcx,
                    obligation.param_env,
                    data.projection_ty,
                    obligation.cause.clone(),
                    0,
                    &mut obligations
                );

                debug!("report_projection_error obligation.cause={:?} obligation.param_env={:?}",
                       obligation.cause, obligation.param_env);

                debug!("report_projection_error normalized_ty={:?} data.ty={:?}",
                       normalized_ty, data.ty);

                let is_normalized_ty_expected = match &obligation.cause.code {
                    ObligationCauseCode::ItemObligation(_) |
                    ObligationCauseCode::BindingObligation(_, _) |
                    ObligationCauseCode::ObjectCastObligation(_) => false,
                    _ => true,
                };

                if let Err(error) = self.at(&obligation.cause, obligation.param_env)
                    .eq_exp(is_normalized_ty_expected, normalized_ty, data.ty)
                {
                    values = Some(infer::ValuePairs::Types(
                        ExpectedFound::new(is_normalized_ty_expected, normalized_ty, data.ty)));

                    err_buf = error;
                    err = &err_buf;
                }
            }

            let msg = format!("type mismatch resolving `{}`", predicate);
            let error_id = (
                DiagnosticMessageId::ErrorId(271),
                Some(obligation.cause.span),
                msg,
            );
            let fresh = self.tcx.sess.one_time_diagnostics.borrow_mut().insert(error_id);
            if fresh {
                let mut diag = struct_span_err!(
                    self.tcx.sess,
                    obligation.cause.span,
                    E0271,
                    "type mismatch resolving `{}`",
                    predicate
                );
                self.note_type_err(&mut diag, &obligation.cause, None, values, err);
                self.note_obligation_cause(&mut diag, obligation);
                diag.emit();
            }
        });
    }

    fn fuzzy_match_tys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
        /// returns the fuzzy category of a given type, or None
        /// if the type can be equated to any type.
        fn type_category(t: Ty<'_>) -> Option<u32> {
            match t.kind {
                ty::Bool => Some(0),
                ty::Char => Some(1),
                ty::Str => Some(2),
                ty::Int(..) | ty::Uint(..) | ty::Infer(ty::IntVar(..)) => Some(3),
                ty::Float(..) | ty::Infer(ty::FloatVar(..)) => Some(4),
                ty::Ref(..) | ty::RawPtr(..) => Some(5),
                ty::Array(..) | ty::Slice(..) => Some(6),
                ty::FnDef(..) | ty::FnPtr(..) => Some(7),
                ty::Dynamic(..) => Some(8),
                ty::Closure(..) => Some(9),
                ty::Tuple(..) => Some(10),
                ty::Projection(..) => Some(11),
                ty::Param(..) => Some(12),
                ty::Opaque(..) => Some(13),
                ty::Never => Some(14),
                ty::Adt(adt, ..) => match adt.adt_kind() {
                    AdtKind::Struct => Some(15),
                    AdtKind::Union => Some(16),
                    AdtKind::Enum => Some(17),
                },
                ty::Generator(..) => Some(18),
                ty::Foreign(..) => Some(19),
                ty::GeneratorWitness(..) => Some(20),
                ty::Placeholder(..) | ty::Bound(..) | ty::Infer(..) | ty::Error => None,
                ty::UnnormalizedProjection(..) => bug!("only used with chalk-engine"),
            }
        }

        match (type_category(a), type_category(b)) {
            (Some(cat_a), Some(cat_b)) => match (&a.kind, &b.kind) {
                (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => def_a == def_b,
                _ => cat_a == cat_b
            },
            // infer and error can be equated to all types
            _ => true
        }
    }

    fn impl_similar_to(&self,
                       trait_ref: ty::PolyTraitRef<'tcx>,
                       obligation: &PredicateObligation<'tcx>)
                       -> Option<DefId>
    {
        let tcx = self.tcx;
        let param_env = obligation.param_env;
        let trait_ref = tcx.erase_late_bound_regions(&trait_ref);
        let trait_self_ty = trait_ref.self_ty();

        let mut self_match_impls = vec![];
        let mut fuzzy_match_impls = vec![];

        self.tcx.for_each_relevant_impl(
            trait_ref.def_id, trait_self_ty, |def_id| {
                let impl_substs = self.fresh_substs_for_item(obligation.cause.span, def_id);
                let impl_trait_ref = tcx
                    .impl_trait_ref(def_id)
                    .unwrap()
                    .subst(tcx, impl_substs);

                let impl_self_ty = impl_trait_ref.self_ty();

                if let Ok(..) = self.can_eq(param_env, trait_self_ty, impl_self_ty) {
                    self_match_impls.push(def_id);

                    if trait_ref.substs.types().skip(1)
                        .zip(impl_trait_ref.substs.types().skip(1))
                        .all(|(u,v)| self.fuzzy_match_tys(u, v))
                    {
                        fuzzy_match_impls.push(def_id);
                    }
                }
            });

        let impl_def_id = if self_match_impls.len() == 1 {
            self_match_impls[0]
        } else if fuzzy_match_impls.len() == 1 {
            fuzzy_match_impls[0]
        } else {
            return None
        };

        if tcx.has_attr(impl_def_id, sym::rustc_on_unimplemented) {
            Some(impl_def_id)
        } else {
            None
        }
    }

    fn describe_generator(&self, body_id: hir::BodyId) -> Option<&'static str> {
        self.tcx.hir().body(body_id).generator_kind.map(|gen_kind| {
            match gen_kind {
                hir::GeneratorKind::Gen => "a generator",
                hir::GeneratorKind::Async(hir::AsyncGeneratorKind::Block) => "an async block",
                hir::GeneratorKind::Async(hir::AsyncGeneratorKind::Fn) => "an async function",
                hir::GeneratorKind::Async(hir::AsyncGeneratorKind::Closure) => "an async closure",
            }
        })
    }

    /// Used to set on_unimplemented's `ItemContext`
    /// to be the enclosing (async) block/function/closure
    fn describe_enclosure(&self, hir_id: hir::HirId) -> Option<&'static str> {
        let hir = &self.tcx.hir();
        let node = hir.find(hir_id)?;
        if let hir::Node::Item(
            hir::Item{kind: hir::ItemKind::Fn(sig, _, body_id), .. }) = &node {
            self.describe_generator(*body_id).or_else(||
                Some(if let hir::FnHeader{ asyncness: hir::IsAsync::Async, .. } = sig.header {
                    "an async function"
                } else {
                    "a function"
                })
            )
        } else if let hir::Node::Expr(hir::Expr {
            kind: hir::ExprKind::Closure(_is_move, _, body_id, _, gen_movability), .. }) = &node {
            self.describe_generator(*body_id).or_else(||
                Some(if gen_movability.is_some() {
                    "an async closure"
                } else {
                    "a closure"
                })
            )
        } else if let hir::Node::Expr(hir::Expr { .. }) = &node {
            let parent_hid = hir.get_parent_node(hir_id);
            if parent_hid != hir_id {
                return self.describe_enclosure(parent_hid);
            } else {
                None
            }
        } else {
            None
        }
    }

    fn on_unimplemented_note(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) -> OnUnimplementedNote {
        let def_id = self.impl_similar_to(trait_ref, obligation)
            .unwrap_or_else(|| trait_ref.def_id());
        let trait_ref = *trait_ref.skip_binder();

        let mut flags = vec![];
        flags.push((sym::item_context,
            self.describe_enclosure(obligation.cause.body_id).map(|s|s.to_owned())));

        match obligation.cause.code {
            ObligationCauseCode::BuiltinDerivedObligation(..) |
            ObligationCauseCode::ImplDerivedObligation(..) => {}
            _ => {
                // this is a "direct", user-specified, rather than derived,
                // obligation.
                flags.push((sym::direct, None));
            }
        }

        if let ObligationCauseCode::ItemObligation(item) = obligation.cause.code {
            // FIXME: maybe also have some way of handling methods
            // from other traits? That would require name resolution,
            // which we might want to be some sort of hygienic.
            //
            // Currently I'm leaving it for what I need for `try`.
            if self.tcx.trait_of_item(item) == Some(trait_ref.def_id) {
                let method = self.tcx.item_name(item);
                flags.push((sym::from_method, None));
                flags.push((sym::from_method, Some(method.to_string())));
            }
        }
        if let Some(t) = self.get_parent_trait_ref(&obligation.cause.code) {
            flags.push((sym::parent_trait, Some(t)));
        }

        if let Some(k) = obligation.cause.span.desugaring_kind() {
            flags.push((sym::from_desugaring, None));
            flags.push((sym::from_desugaring, Some(format!("{:?}", k))));
        }
        let generics = self.tcx.generics_of(def_id);
        let self_ty = trait_ref.self_ty();
        // This is also included through the generics list as `Self`,
        // but the parser won't allow you to use it
        flags.push((sym::_Self, Some(self_ty.to_string())));
        if let Some(def) = self_ty.ty_adt_def() {
            // We also want to be able to select self's original
            // signature with no type arguments resolved
            flags.push((sym::_Self, Some(self.tcx.type_of(def.did).to_string())));
        }

        for param in generics.params.iter() {
            let value = match param.kind {
                GenericParamDefKind::Type { .. } |
                GenericParamDefKind::Const => {
                    trait_ref.substs[param.index as usize].to_string()
                },
                GenericParamDefKind::Lifetime => continue,
            };
            let name = param.name;
            flags.push((name, Some(value)));
        }

        if let Some(true) = self_ty.ty_adt_def().map(|def| def.did.is_local()) {
            flags.push((sym::crate_local, None));
        }

        // Allow targeting all integers using `{integral}`, even if the exact type was resolved
        if self_ty.is_integral() {
            flags.push((sym::_Self, Some("{integral}".to_owned())));
        }

        if let ty::Array(aty, len) = self_ty.kind {
            flags.push((sym::_Self, Some("[]".to_owned())));
            flags.push((sym::_Self, Some(format!("[{}]", aty))));
            if let Some(def) = aty.ty_adt_def() {
                // We also want to be able to select the array's type's original
                // signature with no type arguments resolved
                flags.push((
                    sym::_Self,
                    Some(format!("[{}]", self.tcx.type_of(def.did).to_string())),
                ));
                let tcx = self.tcx;
                if let Some(len) = len.try_eval_usize(tcx, ty::ParamEnv::empty()) {
                    flags.push((
                        sym::_Self,
                        Some(format!("[{}; {}]", self.tcx.type_of(def.did).to_string(), len)),
                    ));
                } else {
                    flags.push((
                        sym::_Self,
                        Some(format!("[{}; _]", self.tcx.type_of(def.did).to_string())),
                    ));
                }
            }
        }

        if let Ok(Some(command)) = OnUnimplementedDirective::of_item(
            self.tcx, trait_ref.def_id, def_id
        ) {
            command.evaluate(self.tcx, trait_ref, &flags[..])
        } else {
            OnUnimplementedNote::empty()
        }
    }

    fn find_similar_impl_candidates(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Vec<ty::TraitRef<'tcx>> {
        let simp = fast_reject::simplify_type(self.tcx, trait_ref.skip_binder().self_ty(), true);
        let all_impls = self.tcx.all_impls(trait_ref.def_id());

        match simp {
            Some(simp) => all_impls.iter().filter_map(|&def_id| {
                let imp = self.tcx.impl_trait_ref(def_id).unwrap();
                let imp_simp = fast_reject::simplify_type(self.tcx, imp.self_ty(), true);
                if let Some(imp_simp) = imp_simp {
                    if simp != imp_simp {
                        return None
                    }
                }

                Some(imp)
            }).collect(),
            None => all_impls.iter().map(|&def_id|
                self.tcx.impl_trait_ref(def_id).unwrap()
            ).collect()
        }
    }

    fn report_similar_impl_candidates(
        &self,
        impl_candidates: Vec<ty::TraitRef<'tcx>>,
        err: &mut DiagnosticBuilder<'_>,
    ) {
        if impl_candidates.is_empty() {
            return;
        }

        let len = impl_candidates.len();
        let end = if impl_candidates.len() <= 5 {
            impl_candidates.len()
        } else {
            4
        };

        let normalize = |candidate| self.tcx.infer_ctxt().enter(|ref infcx| {
            let normalized = infcx
                .at(&ObligationCause::dummy(), ty::ParamEnv::empty())
                .normalize(candidate)
                .ok();
            match normalized {
                Some(normalized) => format!("\n  {:?}", normalized.value),
                None => format!("\n  {:?}", candidate),
            }
        });

        // Sort impl candidates so that ordering is consistent for UI tests.
        let mut normalized_impl_candidates = impl_candidates
            .iter()
            .map(normalize)
            .collect::<Vec<String>>();

        // Sort before taking the `..end` range,
        // because the ordering of `impl_candidates` may not be deterministic:
        // https://github.com/rust-lang/rust/pull/57475#issuecomment-455519507
        normalized_impl_candidates.sort();

        err.help(&format!("the following implementations were found:{}{}",
                          normalized_impl_candidates[..end].join(""),
                          if len > 5 {
                              format!("\nand {} others", len - 4)
                          } else {
                              String::new()
                          }
                          ));
    }

    /// Reports that an overflow has occurred and halts compilation. We
    /// halt compilation unconditionally because it is important that
    /// overflows never be masked -- they basically represent computations
    /// whose result could not be truly determined and thus we can't say
    /// if the program type checks or not -- and they are unusual
    /// occurrences in any case.
    pub fn report_overflow_error<T>(
        &self,
        obligation: &Obligation<'tcx, T>,
        suggest_increasing_limit: bool,
    ) -> !
        where T: fmt::Display + TypeFoldable<'tcx>
    {
        let predicate =
            self.resolve_vars_if_possible(&obligation.predicate);
        let mut err = struct_span_err!(
            self.tcx.sess,
            obligation.cause.span,
            E0275,
            "overflow evaluating the requirement `{}`",
            predicate
        );

        if suggest_increasing_limit {
            self.suggest_new_overflow_limit(&mut err);
        }

        self.note_obligation_cause_code(
            &mut err,
            &obligation.predicate,
            &obligation.cause.code,
            &mut vec![],
        );

        err.emit();
        self.tcx.sess.abort_if_errors();
        bug!();
    }

    /// Reports that a cycle was detected which led to overflow and halts
    /// compilation. This is equivalent to `report_overflow_error` except
    /// that we can give a more helpful error message (and, in particular,
    /// we do not suggest increasing the overflow limit, which is not
    /// going to help).
    pub fn report_overflow_error_cycle(&self, cycle: &[PredicateObligation<'tcx>]) -> ! {
        let cycle = self.resolve_vars_if_possible(&cycle.to_owned());
        assert!(cycle.len() > 0);

        debug!("report_overflow_error_cycle: cycle={:?}", cycle);

        self.report_overflow_error(&cycle[0], false);
    }

    pub fn report_extra_impl_obligation(&self,
                                        error_span: Span,
                                        item_name: ast::Name,
                                        _impl_item_def_id: DefId,
                                        trait_item_def_id: DefId,
                                        requirement: &dyn fmt::Display)
                                        -> DiagnosticBuilder<'tcx>
    {
        let msg = "impl has stricter requirements than trait";
        let sp = self.tcx.sess.source_map().def_span(error_span);

        let mut err = struct_span_err!(self.tcx.sess, sp, E0276, "{}", msg);

        if let Some(trait_item_span) = self.tcx.hir().span_if_local(trait_item_def_id) {
            let span = self.tcx.sess.source_map().def_span(trait_item_span);
            err.span_label(span, format!("definition of `{}` from trait", item_name));
        }

        err.span_label(sp, format!("impl has extra requirement {}", requirement));

        err
    }


    /// Gets the parent trait chain start
    fn get_parent_trait_ref(&self, code: &ObligationCauseCode<'tcx>) -> Option<String> {
        match code {
            &ObligationCauseCode::BuiltinDerivedObligation(ref data) => {
                let parent_trait_ref = self.resolve_vars_if_possible(
                    &data.parent_trait_ref);
                match self.get_parent_trait_ref(&data.parent_code) {
                    Some(t) => Some(t),
                    None => Some(parent_trait_ref.skip_binder().self_ty().to_string()),
                }
            }
            _ => None,
        }
    }

    pub fn report_selection_error(
        &self,
        obligation: &PredicateObligation<'tcx>,
        error: &SelectionError<'tcx>,
        fallback_has_occurred: bool,
        points_at_arg: bool,
    ) {
        let span = obligation.cause.span;

        let mut err = match *error {
            SelectionError::Unimplemented => {
                if let ObligationCauseCode::CompareImplMethodObligation {
                    item_name, impl_item_def_id, trait_item_def_id,
                } = obligation.cause.code {
                    self.report_extra_impl_obligation(
                        span,
                        item_name,
                        impl_item_def_id,
                        trait_item_def_id,
                        &format!("`{}`", obligation.predicate))
                        .emit();
                    return;
                }
                match obligation.predicate {
                    ty::Predicate::Trait(ref trait_predicate) => {
                        let trait_predicate = self.resolve_vars_if_possible(trait_predicate);

                        if self.tcx.sess.has_errors() && trait_predicate.references_error() {
                            return;
                        }
                        let trait_ref = trait_predicate.to_poly_trait_ref();
                        let (
                            post_message,
                            pre_message,
                        ) = self.get_parent_trait_ref(&obligation.cause.code)
                            .map(|t| (format!(" in `{}`", t), format!("within `{}`, ", t)))
                            .unwrap_or_default();

                        let OnUnimplementedNote {
                            message,
                            label,
                            note,
                        } = self.on_unimplemented_note(trait_ref, obligation);
                        let have_alt_message = message.is_some() || label.is_some();
                        let is_try = self.tcx.sess.source_map().span_to_snippet(span)
                            .map(|s| &s == "?")
                            .unwrap_or(false);
                        let is_from = format!("{}", trait_ref).starts_with("std::convert::From<");
                        let (message, note) = if is_try && is_from {
                            (Some(format!(
                                "`?` couldn't convert the error to `{}`",
                                trait_ref.self_ty(),
                            )), Some(
                                "the question mark operation (`?`) implicitly performs a \
                                 conversion on the error value using the `From` trait".to_owned()
                            ))
                        } else {
                            (message, note)
                        };

                        let mut err = struct_span_err!(
                            self.tcx.sess,
                            span,
                            E0277,
                            "{}",
                            message.unwrap_or_else(|| format!(
                                "the trait bound `{}` is not satisfied{}",
                                trait_ref.to_predicate(),
                                post_message,
                            )));

                        let explanation =
                            if obligation.cause.code == ObligationCauseCode::MainFunctionType {
                                "consider using `()`, or a `Result`".to_owned()
                            } else {
                                format!(
                                    "{}the trait `{}` is not implemented for `{}`",
                                    pre_message,
                                    trait_ref,
                                    trait_ref.self_ty(),
                                )
                            };

                        if self.suggest_add_reference_to_arg(
                            &obligation,
                            &mut err,
                            &trait_ref,
                            points_at_arg,
                            have_alt_message,
                        ) {
                            self.note_obligation_cause(&mut err, obligation);
                            err.emit();
                            return;
                        }
                        if let Some(ref s) = label {
                            // If it has a custom `#[rustc_on_unimplemented]`
                            // error message, let's display it as the label!
                            err.span_label(span, s.as_str());
                            err.help(&explanation);
                        } else {
                            err.span_label(span, explanation);
                        }
                        if let Some(ref s) = note {
                            // If it has a custom `#[rustc_on_unimplemented]` note, let's display it
                            err.note(s.as_str());
                        }

                        self.suggest_borrow_on_unsized_slice(&obligation.cause.code, &mut err);
                        self.suggest_fn_call(&obligation, &mut err, &trait_ref, points_at_arg);
                        self.suggest_remove_reference(&obligation, &mut err, &trait_ref);
                        self.suggest_semicolon_removal(&obligation, &mut err, span, &trait_ref);
                        self.note_version_mismatch(&mut err, &trait_ref);

                        // Try to report a help message
                        if !trait_ref.has_infer_types() &&
                            self.predicate_can_apply(obligation.param_env, trait_ref) {
                            // If a where-clause may be useful, remind the
                            // user that they can add it.
                            //
                            // don't display an on-unimplemented note, as
                            // these notes will often be of the form
                            //     "the type `T` can't be frobnicated"
                            // which is somewhat confusing.
                            self.suggest_restricting_param_bound(
                                &mut err,
                                &trait_ref,
                                obligation.cause.body_id,
                            );
                        } else {
                            if !have_alt_message {
                                // Can't show anything else useful, try to find similar impls.
                                let impl_candidates = self.find_similar_impl_candidates(trait_ref);
                                self.report_similar_impl_candidates(impl_candidates, &mut err);
                            }
                            self.suggest_change_mut(
                                &obligation,
                                &mut err,
                                &trait_ref,
                                points_at_arg,
                            );
                        }

                        // If this error is due to `!: Trait` not implemented but `(): Trait` is
                        // implemented, and fallback has occurred, then it could be due to a
                        // variable that used to fallback to `()` now falling back to `!`. Issue a
                        // note informing about the change in behaviour.
                        if trait_predicate.skip_binder().self_ty().is_never()
                            && fallback_has_occurred
                        {
                            let predicate = trait_predicate.map_bound(|mut trait_pred| {
                                trait_pred.trait_ref.substs = self.tcx.mk_substs_trait(
                                    self.tcx.mk_unit(),
                                    &trait_pred.trait_ref.substs[1..],
                                );
                                trait_pred
                            });
                            let unit_obligation = Obligation {
                                predicate: ty::Predicate::Trait(predicate),
                                .. obligation.clone()
                            };
                            if self.predicate_may_hold(&unit_obligation) {
                                err.note("the trait is implemented for `()`. \
                                         Possibly this error has been caused by changes to \
                                         Rust's type-inference algorithm \
                                         (see: https://github.com/rust-lang/rust/issues/48950 \
                                         for more info). Consider whether you meant to use the \
                                         type `()` here instead.");
                            }
                        }

                        err
                    }

                    ty::Predicate::Subtype(ref predicate) => {
                        // Errors for Subtype predicates show up as
                        // `FulfillmentErrorCode::CodeSubtypeError`,
                        // not selection error.
                        span_bug!(span, "subtype requirement gave wrong error: `{:?}`", predicate)
                    }

                    ty::Predicate::RegionOutlives(ref predicate) => {
                        let predicate = self.resolve_vars_if_possible(predicate);
                        let err = self.region_outlives_predicate(&obligation.cause,
                                                                 &predicate).err().unwrap();
                        struct_span_err!(
                            self.tcx.sess, span, E0279,
                            "the requirement `{}` is not satisfied (`{}`)",
                            predicate, err,
                        )
                    }

                    ty::Predicate::Projection(..) | ty::Predicate::TypeOutlives(..) => {
                        let predicate =
                            self.resolve_vars_if_possible(&obligation.predicate);
                        struct_span_err!(self.tcx.sess, span, E0280,
                            "the requirement `{}` is not satisfied",
                            predicate)
                    }

                    ty::Predicate::ObjectSafe(trait_def_id) => {
                        let violations = self.tcx.object_safety_violations(trait_def_id);
                        self.tcx.report_object_safety_error(
                            span,
                            trait_def_id,
                            violations,
                        )
                    }

                    ty::Predicate::ClosureKind(closure_def_id, closure_substs, kind) => {
                        let found_kind = self.closure_kind(closure_def_id, closure_substs).unwrap();
                        let closure_span = self.tcx.sess.source_map()
                            .def_span(self.tcx.hir().span_if_local(closure_def_id).unwrap());
                        let hir_id = self.tcx.hir().as_local_hir_id(closure_def_id).unwrap();
                        let mut err = struct_span_err!(
                            self.tcx.sess, closure_span, E0525,
                            "expected a closure that implements the `{}` trait, \
                             but this closure only implements `{}`",
                            kind,
                            found_kind);

                        err.span_label(
                            closure_span,
                            format!("this closure implements `{}`, not `{}`", found_kind, kind));
                        err.span_label(
                            obligation.cause.span,
                            format!("the requirement to implement `{}` derives from here", kind));

                        // Additional context information explaining why the closure only implements
                        // a particular trait.
                        if let Some(tables) = self.in_progress_tables {
                            let tables = tables.borrow();
                            match (found_kind, tables.closure_kind_origins().get(hir_id)) {
                                (ty::ClosureKind::FnOnce, Some((span, name))) => {
                                    err.span_label(*span, format!(
                                        "closure is `FnOnce` because it moves the \
                                         variable `{}` out of its environment", name));
                                },
                                (ty::ClosureKind::FnMut, Some((span, name))) => {
                                    err.span_label(*span, format!(
                                        "closure is `FnMut` because it mutates the \
                                         variable `{}` here", name));
                                },
                                _ => {}
                            }
                        }

                        err.emit();
                        return;
                    }

                    ty::Predicate::WellFormed(ty) => {
                        if !self.tcx.sess.opts.debugging_opts.chalk {
                            // WF predicates cannot themselves make
                            // errors. They can only block due to
                            // ambiguity; otherwise, they always
                            // degenerate into other obligations
                            // (which may fail).
                            span_bug!(span, "WF predicate not satisfied for {:?}", ty);
                        } else {
                            // FIXME: we'll need a better message which takes into account
                            // which bounds actually failed to hold.
                            self.tcx.sess.struct_span_err(
                                span,
                                &format!("the type `{}` is not well-formed (chalk)", ty)
                            )
                        }
                    }

                    ty::Predicate::ConstEvaluatable(..) => {
                        // Errors for `ConstEvaluatable` predicates show up as
                        // `SelectionError::ConstEvalFailure`,
                        // not `Unimplemented`.
                        span_bug!(span,
                            "const-evaluatable requirement gave wrong error: `{:?}`", obligation)
                    }
                }
            }

            OutputTypeParameterMismatch(ref found_trait_ref, ref expected_trait_ref, _) => {
                let found_trait_ref = self.resolve_vars_if_possible(&*found_trait_ref);
                let expected_trait_ref = self.resolve_vars_if_possible(&*expected_trait_ref);

                if expected_trait_ref.self_ty().references_error() {
                    return;
                }

                let found_trait_ty = found_trait_ref.self_ty();

                let found_did = match found_trait_ty.kind {
                    ty::Closure(did, _) | ty::Foreign(did) | ty::FnDef(did, _) => Some(did),
                    ty::Adt(def, _) => Some(def.did),
                    _ => None,
                };

                let found_span = found_did.and_then(|did|
                    self.tcx.hir().span_if_local(did)
                ).map(|sp| self.tcx.sess.source_map().def_span(sp)); // the sp could be an fn def

                if self.reported_closure_mismatch.borrow().contains(&(span, found_span)) {
                    // We check closures twice, with obligations flowing in different directions,
                    // but we want to complain about them only once.
                    return;
                }

                self.reported_closure_mismatch.borrow_mut().insert((span, found_span));

                let found = match found_trait_ref.skip_binder().substs.type_at(1).kind {
                    ty::Tuple(ref tys) => vec![ArgKind::empty(); tys.len()],
                    _ => vec![ArgKind::empty()],
                };

                let expected_ty = expected_trait_ref.skip_binder().substs.type_at(1);
                let expected = match expected_ty.kind {
                    ty::Tuple(ref tys) => tys.iter()
                        .map(|t| ArgKind::from_expected_ty(t.expect_ty(), Some(span))).collect(),
                    _ => vec![ArgKind::Arg("_".to_owned(), expected_ty.to_string())],
                };

                if found.len() == expected.len() {
                    self.report_closure_arg_mismatch(span,
                                                     found_span,
                                                     found_trait_ref,
                                                     expected_trait_ref)
                } else {
                    let (closure_span, found) = found_did
                        .and_then(|did| self.tcx.hir().get_if_local(did))
                        .map(|node| {
                            let (found_span, found) = self.get_fn_like_arguments(node);
                            (Some(found_span), found)
                        }).unwrap_or((found_span, found));

                    self.report_arg_count_mismatch(span,
                                                   closure_span,
                                                   expected,
                                                   found,
                                                   found_trait_ty.is_closure())
                }
            }

            TraitNotObjectSafe(did) => {
                let violations = self.tcx.object_safety_violations(did);
                self.tcx.report_object_safety_error(span, did, violations)
            }

            // already reported in the query
            ConstEvalFailure(err) => {
                self.tcx.sess.delay_span_bug(
                    span,
                    &format!("constant in type had an ignored error: {:?}", err),
                );
                return;
            }

            Overflow => {
                bug!("overflow should be handled before the `report_selection_error` path");
            }
        };

        self.note_obligation_cause(&mut err, obligation);

        err.emit();
    }

    /// If the `Self` type of the unsatisfied trait `trait_ref` implements a trait
    /// with the same path as `trait_ref`, a help message about
    /// a probable version mismatch is added to `err`
    fn note_version_mismatch(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: &ty::PolyTraitRef<'tcx>,
    ) {
        let get_trait_impl = |trait_def_id| {
            let mut trait_impl = None;
            self.tcx.for_each_relevant_impl(trait_def_id, trait_ref.self_ty(), |impl_def_id| {
                if trait_impl.is_none() {
                    trait_impl = Some(impl_def_id);
                }
            });
            trait_impl
        };
        let required_trait_path = self.tcx.def_path_str(trait_ref.def_id());
        let all_traits = self.tcx.all_traits(LOCAL_CRATE);
        let traits_with_same_path: std::collections::BTreeSet<_> = all_traits
            .iter()
            .filter(|trait_def_id| **trait_def_id != trait_ref.def_id())
            .filter(|trait_def_id| self.tcx.def_path_str(**trait_def_id) == required_trait_path)
            .collect();
        for trait_with_same_path in traits_with_same_path {
            if let Some(impl_def_id) = get_trait_impl(*trait_with_same_path) {
                let impl_span = self.tcx.def_span(impl_def_id);
                err.span_help(impl_span, "trait impl with same name found");
                let trait_crate = self.tcx.crate_name(trait_with_same_path.krate);
                let crate_msg = format!(
                    "Perhaps two different versions of crate `{}` are being used?",
                    trait_crate
                );
                err.note(&crate_msg);
            }
        }
    }
    fn suggest_restricting_param_bound(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: &ty::PolyTraitRef<'_>,
        body_id: hir::HirId,
    ) {
        let self_ty = trait_ref.self_ty();
        let (param_ty, projection) = match &self_ty.kind {
            ty::Param(_) => (true, None),
            ty::Projection(projection) => (false, Some(projection)),
            _ => return,
        };

        let mut suggest_restriction = |generics: &hir::Generics, msg| {
            let span = generics.where_clause.span_for_predicates_or_empty_place();
            if !span.from_expansion() && span.desugaring_kind().is_none() {
                err.span_suggestion(
                    generics.where_clause.span_for_predicates_or_empty_place().shrink_to_hi(),
                    &format!("consider further restricting {}", msg),
                    format!(
                        "{} {} ",
                        if !generics.where_clause.predicates.is_empty() {
                            ","
                        } else {
                            " where"
                        },
                        trait_ref.to_predicate(),
                    ),
                    Applicability::MachineApplicable,
                );
            }
        };

        // FIXME: Add check for trait bound that is already present, particularly `?Sized` so we
        //        don't suggest `T: Sized + ?Sized`.
        let mut hir_id = body_id;
        while let Some(node) = self.tcx.hir().find(hir_id) {
            match node {
                hir::Node::TraitItem(hir::TraitItem {
                    generics,
                    kind: hir::TraitItemKind::Method(..), ..
                }) if param_ty && self_ty == self.tcx.types.self_param => {
                    // Restricting `Self` for a single method.
                    suggest_restriction(&generics, "`Self`");
                    return;
                }

                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Fn(_, generics, _), ..
                }) |
                hir::Node::TraitItem(hir::TraitItem {
                    generics,
                    kind: hir::TraitItemKind::Method(..), ..
                }) |
                hir::Node::ImplItem(hir::ImplItem {
                    generics,
                    kind: hir::ImplItemKind::Method(..), ..
                }) |
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Trait(_, _, generics, _, _), ..
                }) |
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Impl(_, _, _, generics, ..), ..
                }) if projection.is_some() => {
                    // Missing associated type bound.
                    suggest_restriction(&generics, "the associated type");
                    return;
                }

                hir::Node::Item(hir::Item { kind: hir::ItemKind::Struct(_, generics), span, .. }) |
                hir::Node::Item(hir::Item { kind: hir::ItemKind::Enum(_, generics), span, .. }) |
                hir::Node::Item(hir::Item { kind: hir::ItemKind::Union(_, generics), span, .. }) |
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Trait(_, _, generics, ..), span, ..
                }) |
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Impl(_, _, _, generics, ..), span, ..
                }) |
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Fn(_, generics, _), span, ..
                }) |
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::TyAlias(_, generics), span, ..
                }) |
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::TraitAlias(generics, _), span, ..
                }) |
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::OpaqueTy(hir::OpaqueTy { generics, .. }), span, ..
                }) |
                hir::Node::TraitItem(hir::TraitItem { generics, span, .. }) |
                hir::Node::ImplItem(hir::ImplItem { generics, span, .. })
                if param_ty => {
                    // Missing generic type parameter bound.
                    let restrict_msg = "consider further restricting this bound";
                    let param_name = self_ty.to_string();
                    for param in generics.params.iter().filter(|p| {
                        p.name.ident().as_str() == param_name
                    }) {
                        if param_name.starts_with("impl ") {
                            // `impl Trait` in argument:
                            // `fn foo(x: impl Trait) {}` → `fn foo(t: impl Trait + Trait2) {}`
                            err.span_suggestion(
                                param.span,
                                restrict_msg,
                                // `impl CurrentTrait + MissingTrait`
                                format!("{} + {}", param.name.ident(), trait_ref),
                                Applicability::MachineApplicable,
                            );
                        } else if generics.where_clause.predicates.is_empty() &&
                                param.bounds.is_empty()
                        {
                            // If there are no bounds whatsoever, suggest adding a constraint
                            // to the type parameter:
                            // `fn foo<T>(t: T) {}` → `fn foo<T: Trait>(t: T) {}`
                            err.span_suggestion(
                                param.span,
                                "consider restricting this bound",
                                format!("{}", trait_ref.to_predicate()),
                                Applicability::MachineApplicable,
                            );
                        } else if !generics.where_clause.predicates.is_empty() {
                            // There is a `where` clause, so suggest expanding it:
                            // `fn foo<T>(t: T) where T: Debug {}` →
                            // `fn foo<T>(t: T) where T: Debug, T: Trait {}`
                            err.span_suggestion(
                                generics.where_clause.span().unwrap().shrink_to_hi(),
                                &format!(
                                    "consider further restricting type parameter `{}`",
                                    param_name,
                                ),
                                format!(", {}", trait_ref.to_predicate()),
                                Applicability::MachineApplicable,
                            );
                        } else {
                            // If there is no `where` clause lean towards constraining to the
                            // type parameter:
                            // `fn foo<X: Bar, T>(t: T, x: X) {}` → `fn foo<T: Trait>(t: T) {}`
                            // `fn foo<T: Bar>(t: T) {}` → `fn foo<T: Bar + Trait>(t: T) {}`
                            let sp = param.span.with_hi(span.hi());
                            let span = self.tcx.sess.source_map()
                                .span_through_char(sp, ':');
                            if sp != param.span && sp != span {
                                // Only suggest if we have high certainty that the span
                                // covers the colon in `foo<T: Trait>`.
                                err.span_suggestion(span, restrict_msg, format!(
                                    "{} + ",
                                    trait_ref.to_predicate(),
                                ), Applicability::MachineApplicable);
                            } else {
                                err.span_label(param.span, &format!(
                                    "consider adding a `where {}` bound",
                                    trait_ref.to_predicate(),
                                ));
                            }
                        }
                        return;
                    }
                }

                hir::Node::Crate => return,

                _ => {}
            }

            hir_id = self.tcx.hir().get_parent_item(hir_id);
        }
    }

    /// When encountering an assignment of an unsized trait, like `let x = ""[..];`, provide a
    /// suggestion to borrow the initializer in order to use have a slice instead.
    fn suggest_borrow_on_unsized_slice(
        &self,
        code: &ObligationCauseCode<'tcx>,
        err: &mut DiagnosticBuilder<'tcx>,
    ) {
        if let &ObligationCauseCode::VariableType(hir_id) = code {
            let parent_node = self.tcx.hir().get_parent_node(hir_id);
            if let Some(Node::Local(ref local)) = self.tcx.hir().find(parent_node) {
                if let Some(ref expr) = local.init {
                    if let hir::ExprKind::Index(_, _) = expr.kind {
                        if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(expr.span) {
                            err.span_suggestion(
                                expr.span,
                                "consider borrowing here",
                                format!("&{}", snippet),
                                Applicability::MachineApplicable
                            );
                        }
                    }
                }
            }
        }
    }

    fn mk_obligation_for_def_id(
        &self,
        def_id: DefId,
        output_ty: Ty<'tcx>,
        cause: ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> PredicateObligation<'tcx> {
        let new_trait_ref = ty::TraitRef {
            def_id,
            substs: self.tcx.mk_substs_trait(output_ty, &[]),
        };
        Obligation::new(cause, param_env, new_trait_ref.to_predicate())
    }

    /// Given a closure's `DefId`, return the given name of the closure.
    ///
    /// This doesn't account for reassignments, but it's only used for suggestions.
    fn get_closure_name(
        &self,
        def_id: DefId,
        err: &mut DiagnosticBuilder<'_>,
        msg: &str,
    ) -> Option<String> {
        let get_name = |err: &mut DiagnosticBuilder<'_>, kind: &hir::PatKind| -> Option<String> {
            // Get the local name of this closure. This can be inaccurate because
            // of the possibility of reassignment, but this should be good enough.
            match &kind {
                hir::PatKind::Binding(hir::BindingAnnotation::Unannotated, _, name, None) => {
                    Some(format!("{}", name))
                }
                _ => {
                    err.note(&msg);
                    None
                }
            }
        };

        let hir = self.tcx.hir();
        let hir_id = hir.as_local_hir_id(def_id)?;
        let parent_node = hir.get_parent_node(hir_id);
        match hir.find(parent_node) {
            Some(hir::Node::Stmt(hir::Stmt {
                kind: hir::StmtKind::Local(local), ..
            })) => get_name(err, &local.pat.kind),
            // Different to previous arm because one is `&hir::Local` and the other
            // is `P<hir::Local>`.
            Some(hir::Node::Local(local)) => get_name(err, &local.pat.kind),
            _ => return None,
        }
    }

    /// We tried to apply the bound to an `fn` or closure. Check whether calling it would
    /// evaluate to a type that *would* satisfy the trait binding. If it would, suggest calling
    /// it: `bar(foo)` → `bar(foo())`. This case is *very* likely to be hit if `foo` is `async`.
    fn suggest_fn_call(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
        trait_ref: &ty::Binder<ty::TraitRef<'tcx>>,
        points_at_arg: bool,
    ) {
        let self_ty = trait_ref.self_ty();
        let (def_id, output_ty, callable) = match self_ty.kind {
            ty::Closure(def_id, substs) => {
                (def_id, self.closure_sig(def_id, substs).output(), "closure")
            }
            ty::FnDef(def_id, _) => {
                (def_id, self_ty.fn_sig(self.tcx).output(), "function")
            }
            _ => return,
        };
        let msg = format!("use parentheses to call the {}", callable);

        let obligation = self.mk_obligation_for_def_id(
            trait_ref.def_id(),
            output_ty.skip_binder(),
            obligation.cause.clone(),
            obligation.param_env,
        );

        match self.evaluate_obligation(&obligation) {
            Ok(EvaluationResult::EvaluatedToOk) |
            Ok(EvaluationResult::EvaluatedToOkModuloRegions) |
            Ok(EvaluationResult::EvaluatedToAmbig) => {}
            _ => return,
        }
        let hir = self.tcx.hir();
        // Get the name of the callable and the arguments to be used in the suggestion.
        let snippet = match hir.get_if_local(def_id) {
            Some(hir::Node::Expr(hir::Expr {
                kind: hir::ExprKind::Closure(_, decl, _, span, ..),
                ..
            })) => {
                err.span_label(*span, "consider calling this closure");
                let name = match self.get_closure_name(def_id, err, &msg) {
                    Some(name) => name,
                    None => return,
                };
                let args = decl.inputs.iter()
                    .map(|_| "_")
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", name, args)
            }
            Some(hir::Node::Item(hir::Item {
                ident,
                kind: hir::ItemKind::Fn(.., body_id),
                ..
            })) => {
                err.span_label(ident.span, "consider calling this function");
                let body = hir.body(*body_id);
                let args = body.params.iter()
                    .map(|arg| match &arg.pat.kind {
                        hir::PatKind::Binding(_, _, ident, None)
                        // FIXME: provide a better suggestion when encountering `SelfLower`, it
                        // should suggest a method call.
                        if ident.name != kw::SelfLower => ident.to_string(),
                        _ => "_".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", ident, args)
            }
            _ => return,
        };
        if points_at_arg {
            // When the obligation error has been ensured to have been caused by
            // an argument, the `obligation.cause.span` points at the expression
            // of the argument, so we can provide a suggestion. This is signaled
            // by `points_at_arg`. Otherwise, we give a more general note.
            err.span_suggestion(
                obligation.cause.span,
                &msg,
                snippet,
                Applicability::HasPlaceholders,
            );
        } else {
            err.help(&format!("{}: `{}`", msg, snippet));
        }
    }

    fn suggest_add_reference_to_arg(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'tcx>,
        trait_ref: &ty::Binder<ty::TraitRef<'tcx>>,
        points_at_arg: bool,
        has_custom_message: bool,
    ) -> bool {
        if !points_at_arg {
            return false;
        }

        let span = obligation.cause.span;
        let param_env = obligation.param_env;
        let trait_ref = trait_ref.skip_binder();

        if let ObligationCauseCode::ImplDerivedObligation(obligation) = &obligation.cause.code {
            // Try to apply the original trait binding obligation by borrowing.
            let self_ty = trait_ref.self_ty();
            let found = self_ty.to_string();
            let new_self_ty = self.tcx.mk_imm_ref(self.tcx.lifetimes.re_static, self_ty);
            let substs = self.tcx.mk_substs_trait(new_self_ty, &[]);
            let new_trait_ref = ty::TraitRef::new(obligation.parent_trait_ref.def_id(), substs);
            let new_obligation = Obligation::new(
                ObligationCause::dummy(),
                param_env,
                new_trait_ref.to_predicate(),
            );
            if self.predicate_must_hold_modulo_regions(&new_obligation) {
                if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                    // We have a very specific type of error, where just borrowing this argument
                    // might solve the problem. In cases like this, the important part is the
                    // original type obligation, not the last one that failed, which is arbitrary.
                    // Because of this, we modify the error to refer to the original obligation and
                    // return early in the caller.
                    let msg = format!(
                        "the trait bound `{}: {}` is not satisfied",
                        found,
                        obligation.parent_trait_ref.skip_binder(),
                    );
                    if has_custom_message {
                        err.note(&msg);
                    } else {
                        err.message = vec![(msg, Style::NoStyle)];
                    }
                    if snippet.starts_with('&') {
                        // This is already a literal borrow and the obligation is failing
                        // somewhere else in the obligation chain. Do not suggest non-sense.
                        return false;
                    }
                    err.span_label(span, &format!(
                        "expected an implementor of trait `{}`",
                        obligation.parent_trait_ref.skip_binder(),
                    ));
                    err.span_suggestion(
                        span,
                        "consider borrowing here",
                        format!("&{}", snippet),
                        Applicability::MaybeIncorrect,
                    );
                    return true;
                }
            }
        }
        false
    }

    /// Whenever references are used by mistake, like `for (i, e) in &vec.iter().enumerate()`,
    /// suggest removing these references until we reach a type that implements the trait.
    fn suggest_remove_reference(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'tcx>,
        trait_ref: &ty::Binder<ty::TraitRef<'tcx>>,
    ) {
        let trait_ref = trait_ref.skip_binder();
        let span = obligation.cause.span;

        if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
            let refs_number = snippet.chars()
                .filter(|c| !c.is_whitespace())
                .take_while(|c| *c == '&')
                .count();
            if let Some('\'') = snippet.chars()
                .filter(|c| !c.is_whitespace())
                .skip(refs_number)
                .next()
            { // Do not suggest removal of borrow from type arguments.
                return;
            }

            let mut trait_type = trait_ref.self_ty();

            for refs_remaining in 0..refs_number {
                if let ty::Ref(_, t_type, _) = trait_type.kind {
                    trait_type = t_type;

                    let new_obligation = self.mk_obligation_for_def_id(
                        trait_ref.def_id,
                        trait_type,
                        ObligationCause::dummy(),
                        obligation.param_env,
                    );

                    if self.predicate_may_hold(&new_obligation) {
                        let sp = self.tcx.sess.source_map()
                            .span_take_while(span, |c| c.is_whitespace() || *c == '&');

                        let remove_refs = refs_remaining + 1;
                        let format_str = format!("consider removing {} leading `&`-references",
                                                 remove_refs);

                        err.span_suggestion_short(
                            sp, &format_str, String::new(), Applicability::MachineApplicable
                        );
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    /// Check if the trait bound is implemented for a different mutability and note it in the
    /// final error.
    fn suggest_change_mut(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'tcx>,
        trait_ref: &ty::Binder<ty::TraitRef<'tcx>>,
        points_at_arg: bool,
    ) {
        let span = obligation.cause.span;
        if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
            let refs_number = snippet.chars()
                .filter(|c| !c.is_whitespace())
                .take_while(|c| *c == '&')
                .count();
            if let Some('\'') = snippet.chars()
                .filter(|c| !c.is_whitespace())
                .skip(refs_number)
                .next()
            { // Do not suggest removal of borrow from type arguments.
                return;
            }
            let trait_ref = self.resolve_vars_if_possible(trait_ref);
            if trait_ref.has_infer_types() {
                // Do not ICE while trying to find if a reborrow would succeed on a trait with
                // unresolved bindings.
                return;
            }

            if let ty::Ref(region, t_type, mutability) = trait_ref.skip_binder().self_ty().kind {
                let trait_type = match mutability {
                    hir::Mutability::Mutable => self.tcx.mk_imm_ref(region, t_type),
                    hir::Mutability::Immutable => self.tcx.mk_mut_ref(region, t_type),
                };

                let new_obligation = self.mk_obligation_for_def_id(
                    trait_ref.skip_binder().def_id,
                    trait_type,
                    ObligationCause::dummy(),
                    obligation.param_env,
                );

                if self.evaluate_obligation_no_overflow(
                    &new_obligation,
                ).must_apply_modulo_regions() {
                    let sp = self.tcx.sess.source_map()
                        .span_take_while(span, |c| c.is_whitespace() || *c == '&');
                    if points_at_arg &&
                        mutability == hir::Mutability::Immutable &&
                        refs_number > 0
                    {
                        err.span_suggestion(
                            sp,
                            "consider changing this borrow's mutability",
                            "&mut ".to_string(),
                            Applicability::MachineApplicable,
                        );
                    } else {
                        err.note(&format!(
                            "`{}` is implemented for `{:?}`, but not for `{:?}`",
                            trait_ref,
                            trait_type,
                            trait_ref.skip_binder().self_ty(),
                        ));
                    }
                }
            }
        }
    }

    fn suggest_semicolon_removal(
        &self,
        obligation: &PredicateObligation<'tcx>,
        err: &mut DiagnosticBuilder<'tcx>,
        span: Span,
        trait_ref: &ty::Binder<ty::TraitRef<'tcx>>,
    ) {
        let hir = self.tcx.hir();
        let parent_node = hir.get_parent_node(obligation.cause.body_id);
        let node = hir.find(parent_node);
        if let Some(hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Fn(sig, _, body_id),
            ..
        })) = node {
            let body = hir.body(*body_id);
            if let hir::ExprKind::Block(blk, _) = &body.value.kind {
                if sig.decl.output.span().overlaps(span) && blk.expr.is_none() &&
                    "()" == &trait_ref.self_ty().to_string()
                {
                    // FIXME(estebank): When encountering a method with a trait
                    // bound not satisfied in the return type with a body that has
                    // no return, suggest removal of semicolon on last statement.
                    // Once that is added, close #54771.
                    if let Some(ref stmt) = blk.stmts.last() {
                        let sp = self.tcx.sess.source_map().end_point(stmt.span);
                        err.span_label(sp, "consider removing this semicolon");
                    }
                }
            }
        }
    }

    /// Given some node representing a fn-like thing in the HIR map,
    /// returns a span and `ArgKind` information that describes the
    /// arguments it expects. This can be supplied to
    /// `report_arg_count_mismatch`.
    pub fn get_fn_like_arguments(&self, node: Node<'_>) -> (Span, Vec<ArgKind>) {
        match node {
            Node::Expr(&hir::Expr {
                kind: hir::ExprKind::Closure(_, ref _decl, id, span, _),
                ..
            }) => {
                (self.tcx.sess.source_map().def_span(span),
                 self.tcx.hir().body(id).params.iter()
                    .map(|arg| {
                        if let hir::Pat {
                            kind: hir::PatKind::Tuple(ref args, _),
                            span,
                            ..
                        } = *arg.pat {
                            ArgKind::Tuple(
                                Some(span),
                                args.iter().map(|pat| {
                                    let snippet = self.tcx.sess.source_map()
                                        .span_to_snippet(pat.span).unwrap();
                                    (snippet, "_".to_owned())
                                }).collect::<Vec<_>>(),
                            )
                        } else {
                            let name = self.tcx.sess.source_map()
                                .span_to_snippet(arg.pat.span).unwrap();
                            ArgKind::Arg(name, "_".to_owned())
                        }
                    })
                    .collect::<Vec<ArgKind>>())
            }
            Node::Item(&hir::Item {
                span,
                kind: hir::ItemKind::Fn(ref sig, ..),
                ..
            }) |
            Node::ImplItem(&hir::ImplItem {
                span,
                kind: hir::ImplItemKind::Method(ref sig, _),
                ..
            }) |
            Node::TraitItem(&hir::TraitItem {
                span,
                kind: hir::TraitItemKind::Method(ref sig, _),
                ..
            }) => {
                (self.tcx.sess.source_map().def_span(span), sig.decl.inputs.iter()
                        .map(|arg| match arg.clone().kind {
                    hir::TyKind::Tup(ref tys) => ArgKind::Tuple(
                        Some(arg.span),
                        vec![("_".to_owned(), "_".to_owned()); tys.len()]
                    ),
                    _ => ArgKind::empty()
                }).collect::<Vec<ArgKind>>())
            }
            Node::Ctor(ref variant_data) => {
                let span = variant_data.ctor_hir_id()
                    .map(|hir_id| self.tcx.hir().span(hir_id))
                    .unwrap_or(DUMMY_SP);
                let span = self.tcx.sess.source_map().def_span(span);

                (span, vec![ArgKind::empty(); variant_data.fields().len()])
            }
            _ => panic!("non-FnLike node found: {:?}", node),
        }
    }

    /// Reports an error when the number of arguments needed by a
    /// trait match doesn't match the number that the expression
    /// provides.
    pub fn report_arg_count_mismatch(
        &self,
        span: Span,
        found_span: Option<Span>,
        expected_args: Vec<ArgKind>,
        found_args: Vec<ArgKind>,
        is_closure: bool,
    ) -> DiagnosticBuilder<'tcx> {
        let kind = if is_closure { "closure" } else { "function" };

        let args_str = |arguments: &[ArgKind], other: &[ArgKind]| {
            let arg_length = arguments.len();
            let distinct = match &other[..] {
                &[ArgKind::Tuple(..)] => true,
                _ => false,
            };
            match (arg_length, arguments.get(0)) {
                (1, Some(&ArgKind::Tuple(_, ref fields))) => {
                    format!("a single {}-tuple as argument", fields.len())
                }
                _ => format!("{} {}argument{}",
                             arg_length,
                             if distinct && arg_length > 1 { "distinct " } else { "" },
                             pluralize!(arg_length))
            }
        };

        let expected_str = args_str(&expected_args, &found_args);
        let found_str = args_str(&found_args, &expected_args);

        let mut err = struct_span_err!(
            self.tcx.sess,
            span,
            E0593,
            "{} is expected to take {}, but it takes {}",
            kind,
            expected_str,
            found_str,
        );

        err.span_label(span, format!("expected {} that takes {}", kind, expected_str));

        if let Some(found_span) = found_span {
            err.span_label(found_span, format!("takes {}", found_str));

            // move |_| { ... }
            // ^^^^^^^^-- def_span
            //
            // move |_| { ... }
            // ^^^^^-- prefix
            let prefix_span = self.tcx.sess.source_map().span_until_non_whitespace(found_span);
            // move |_| { ... }
            //      ^^^-- pipe_span
            let pipe_span = if let Some(span) = found_span.trim_start(prefix_span) {
                span
            } else {
                found_span
            };

            // Suggest to take and ignore the arguments with expected_args_length `_`s if
            // found arguments is empty (assume the user just wants to ignore args in this case).
            // For example, if `expected_args_length` is 2, suggest `|_, _|`.
            if found_args.is_empty() && is_closure {
                let underscores = vec!["_"; expected_args.len()].join(", ");
                err.span_suggestion(
                    pipe_span,
                    &format!(
                        "consider changing the closure to take and ignore the expected argument{}",
                        if expected_args.len() < 2 {
                            ""
                        } else {
                            "s"
                        }
                    ),
                    format!("|{}|", underscores),
                    Applicability::MachineApplicable,
                );
            }

            if let &[ArgKind::Tuple(_, ref fields)] = &found_args[..] {
                if fields.len() == expected_args.len() {
                    let sugg = fields.iter()
                        .map(|(name, _)| name.to_owned())
                        .collect::<Vec<String>>()
                        .join(", ");
                    err.span_suggestion(
                        found_span,
                        "change the closure to take multiple arguments instead of a single tuple",
                        format!("|{}|", sugg),
                        Applicability::MachineApplicable,
                    );
                }
            }
            if let &[ArgKind::Tuple(_, ref fields)] = &expected_args[..] {
                if fields.len() == found_args.len() && is_closure {
                    let sugg = format!(
                        "|({}){}|",
                        found_args.iter()
                            .map(|arg| match arg {
                                ArgKind::Arg(name, _) => name.to_owned(),
                                _ => "_".to_owned(),
                            })
                            .collect::<Vec<String>>()
                            .join(", "),
                        // add type annotations if available
                        if found_args.iter().any(|arg| match arg {
                            ArgKind::Arg(_, ty) => ty != "_",
                            _ => false,
                        }) {
                            format!(": ({})",
                                    fields.iter()
                                        .map(|(_, ty)| ty.to_owned())
                                        .collect::<Vec<String>>()
                                        .join(", "))
                        } else {
                            String::new()
                        },
                    );
                    err.span_suggestion(
                        found_span,
                        "change the closure to accept a tuple instead of individual arguments",
                        sugg,
                        Applicability::MachineApplicable,
                    );
                }
            }
        }

        err
    }

    fn report_closure_arg_mismatch(
        &self,
        span: Span,
        found_span: Option<Span>,
        expected_ref: ty::PolyTraitRef<'tcx>,
        found: ty::PolyTraitRef<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        fn build_fn_sig_string<'tcx>(tcx: TyCtxt<'tcx>, trait_ref: &ty::TraitRef<'tcx>) -> String {
            let inputs = trait_ref.substs.type_at(1);
            let sig = if let ty::Tuple(inputs) = inputs.kind {
                tcx.mk_fn_sig(
                    inputs.iter().map(|k| k.expect_ty()),
                    tcx.mk_ty_infer(ty::TyVar(ty::TyVid { index: 0 })),
                    false,
                    hir::Unsafety::Normal,
                    ::rustc_target::spec::abi::Abi::Rust
                )
            } else {
                tcx.mk_fn_sig(
                    ::std::iter::once(inputs),
                    tcx.mk_ty_infer(ty::TyVar(ty::TyVid { index: 0 })),
                    false,
                    hir::Unsafety::Normal,
                    ::rustc_target::spec::abi::Abi::Rust
                )
            };
            ty::Binder::bind(sig).to_string()
        }

        let argument_is_closure = expected_ref.skip_binder().substs.type_at(0).is_closure();
        let mut err = struct_span_err!(self.tcx.sess, span, E0631,
                                       "type mismatch in {} arguments",
                                       if argument_is_closure { "closure" } else { "function" });

        let found_str = format!(
            "expected signature of `{}`",
            build_fn_sig_string(self.tcx, found.skip_binder())
        );
        err.span_label(span, found_str);

        let found_span = found_span.unwrap_or(span);
        let expected_str = format!(
            "found signature of `{}`",
            build_fn_sig_string(self.tcx, expected_ref.skip_binder())
        );
        err.span_label(found_span, expected_str);

        err
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn recursive_type_with_infinite_size_error(self,
                                                   type_def_id: DefId)
                                                   -> DiagnosticBuilder<'tcx>
    {
        assert!(type_def_id.is_local());
        let span = self.hir().span_if_local(type_def_id).unwrap();
        let span = self.sess.source_map().def_span(span);
        let mut err = struct_span_err!(self.sess, span, E0072,
                                       "recursive type `{}` has infinite size",
                                       self.def_path_str(type_def_id));
        err.span_label(span, "recursive type has infinite size");
        err.help(&format!("insert indirection (e.g., a `Box`, `Rc`, or `&`) \
                           at some point to make `{}` representable",
                          self.def_path_str(type_def_id)));
        err
    }

    pub fn report_object_safety_error(
        self,
        span: Span,
        trait_def_id: DefId,
        violations: Vec<ObjectSafetyViolation>,
    ) -> DiagnosticBuilder<'tcx> {
        let trait_str = self.def_path_str(trait_def_id);
        let span = self.sess.source_map().def_span(span);
        let mut err = struct_span_err!(
            self.sess, span, E0038,
            "the trait `{}` cannot be made into an object",
            trait_str);
        err.span_label(span, format!("the trait `{}` cannot be made into an object", trait_str));

        let mut reported_violations = FxHashSet::default();
        for violation in violations {
            if reported_violations.insert(violation.clone()) {
                match violation.span() {
                    Some(span) => err.span_label(span, violation.error_msg()),
                    None => err.note(&violation.error_msg()),
                };
            }
        }

        if self.sess.trait_methods_not_found.borrow().contains(&span) {
            // Avoid emitting error caused by non-existing method (#58734)
            err.cancel();
        }

        err
    }
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    fn maybe_report_ambiguity(
        &self,
        obligation: &PredicateObligation<'tcx>,
        body_id: Option<hir::BodyId>,
    ) {
        // Unable to successfully determine, probably means
        // insufficient type information, but could mean
        // ambiguous impls. The latter *ought* to be a
        // coherence violation, so we don't report it here.

        let predicate = self.resolve_vars_if_possible(&obligation.predicate);
        let span = obligation.cause.span;

        debug!(
            "maybe_report_ambiguity(predicate={:?}, obligation={:?} body_id={:?}, code={:?})",
            predicate,
            obligation,
            body_id,
            obligation.cause.code,
        );

        // Ambiguity errors are often caused as fallout from earlier
        // errors. So just ignore them if this infcx is tainted.
        if self.is_tainted_by_errors() {
            return;
        }

        match predicate {
            ty::Predicate::Trait(ref data) => {
                let trait_ref = data.to_poly_trait_ref();
                let self_ty = trait_ref.self_ty();
                debug!("self_ty {:?} {:?} trait_ref {:?}", self_ty, self_ty.kind, trait_ref);

                if predicate.references_error() {
                    return;
                }
                // Typically, this ambiguity should only happen if
                // there are unresolved type inference variables
                // (otherwise it would suggest a coherence
                // failure). But given #21974 that is not necessarily
                // the case -- we can have multiple where clauses that
                // are only distinguished by a region, which results
                // in an ambiguity even when all types are fully
                // known, since we don't dispatch based on region
                // relationships.

                // This is kind of a hack: it frequently happens that some earlier
                // error prevents types from being fully inferred, and then we get
                // a bunch of uninteresting errors saying something like "<generic
                // #0> doesn't implement Sized".  It may even be true that we
                // could just skip over all checks where the self-ty is an
                // inference variable, but I was afraid that there might be an
                // inference variable created, registered as an obligation, and
                // then never forced by writeback, and hence by skipping here we'd
                // be ignoring the fact that we don't KNOW the type works
                // out. Though even that would probably be harmless, given that
                // we're only talking about builtin traits, which are known to be
                // inhabited. We used to check for `self.tcx.sess.has_errors()` to
                // avoid inundating the user with unnecessary errors, but we now
                // check upstream for type errors and dont add the obligations to
                // begin with in those cases.
                if
                    self.tcx.lang_items().sized_trait()
                    .map_or(false, |sized_id| sized_id == trait_ref.def_id())
                {
                    self.need_type_info_err(body_id, span, self_ty).emit();
                } else {
                    let mut err = struct_span_err!(
                        self.tcx.sess,
                        span,
                        E0283,
                        "type annotations needed: cannot resolve `{}`",
                        predicate,
                    );
                    self.note_obligation_cause(&mut err, obligation);
                    err.emit();
                }
            }

            ty::Predicate::WellFormed(ty) => {
                // Same hacky approach as above to avoid deluging user
                // with error messages.
                if !ty.references_error() && !self.tcx.sess.has_errors() {
                    self.need_type_info_err(body_id, span, ty).emit();
                }
            }

            ty::Predicate::Subtype(ref data) => {
                if data.references_error() || self.tcx.sess.has_errors() {
                    // no need to overload user in such cases
                } else {
                    let &SubtypePredicate { a_is_expected: _, a, b } = data.skip_binder();
                    // both must be type variables, or the other would've been instantiated
                    assert!(a.is_ty_var() && b.is_ty_var());
                    self.need_type_info_err(body_id,
                                            obligation.cause.span,
                                            a).emit();
                }
            }

            _ => {
                if !self.tcx.sess.has_errors() {
                    let mut err = struct_span_err!(
                        self.tcx.sess,
                        obligation.cause.span,
                        E0284,
                        "type annotations needed: cannot resolve `{}`",
                        predicate,
                    );
                    self.note_obligation_cause(&mut err, obligation);
                    err.emit();
                }
            }
        }
    }

    /// Returns `true` if the trait predicate may apply for *some* assignment
    /// to the type parameters.
    fn predicate_can_apply(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        pred: ty::PolyTraitRef<'tcx>,
    ) -> bool {
        struct ParamToVarFolder<'a, 'tcx> {
            infcx: &'a InferCtxt<'a, 'tcx>,
            var_map: FxHashMap<Ty<'tcx>, Ty<'tcx>>,
        }

        impl<'a, 'tcx> TypeFolder<'tcx> for ParamToVarFolder<'a, 'tcx> {
            fn tcx<'b>(&'b self) -> TyCtxt<'tcx> { self.infcx.tcx }

            fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
                if let ty::Param(ty::ParamTy {name, .. }) = ty.kind {
                    let infcx = self.infcx;
                    self.var_map.entry(ty).or_insert_with(||
                        infcx.next_ty_var(
                            TypeVariableOrigin {
                                kind: TypeVariableOriginKind::TypeParameterDefinition(name),
                                span: DUMMY_SP,
                            }
                        )
                    )
                } else {
                    ty.super_fold_with(self)
                }
            }
        }

        self.probe(|_| {
            let mut selcx = SelectionContext::new(self);

            let cleaned_pred = pred.fold_with(&mut ParamToVarFolder {
                infcx: self,
                var_map: Default::default()
            });

            let cleaned_pred = super::project::normalize(
                &mut selcx,
                param_env,
                ObligationCause::dummy(),
                &cleaned_pred
            ).value;

            let obligation = Obligation::new(
                ObligationCause::dummy(),
                param_env,
                cleaned_pred.to_predicate()
            );

            self.predicate_may_hold(&obligation)
        })
    }

    fn note_obligation_cause(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        obligation: &PredicateObligation<'tcx>,
    ) {
        // First, attempt to add note to this error with an async-await-specific
        // message, and fall back to regular note otherwise.
        if !self.note_obligation_cause_for_async_await(err, obligation) {
            self.note_obligation_cause_code(err, &obligation.predicate, &obligation.cause.code,
                                            &mut vec![]);
        }
    }

    /// Adds an async-await specific note to the diagnostic:
    ///
    /// ```ignore (diagnostic)
    /// note: future does not implement `std::marker::Send` because this value is used across an
    ///       await
    ///   --> $DIR/issue-64130-non-send-future-diags.rs:15:5
    ///    |
    /// LL |     let g = x.lock().unwrap();
    ///    |         - has type `std::sync::MutexGuard<'_, u32>`
    /// LL |     baz().await;
    ///    |     ^^^^^^^^^^^ await occurs here, with `g` maybe used later
    /// LL | }
    ///    | - `g` is later dropped here
    /// ```
    ///
    /// Returns `true` if an async-await specific note was added to the diagnostic.
    fn note_obligation_cause_for_async_await(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        obligation: &PredicateObligation<'tcx>,
    ) -> bool {
        debug!("note_obligation_cause_for_async_await: obligation.predicate={:?} \
                obligation.cause.span={:?}", obligation.predicate, obligation.cause.span);
        let source_map = self.tcx.sess.source_map();

        // Look into the obligation predicate to determine the type in the generator which meant
        // that the predicate was not satisifed.
        let (trait_ref, target_ty) = match obligation.predicate {
            ty::Predicate::Trait(trait_predicate) =>
                (trait_predicate.skip_binder().trait_ref, trait_predicate.skip_binder().self_ty()),
            _ => return false,
        };
        debug!("note_obligation_cause_for_async_await: target_ty={:?}", target_ty);

        // Attempt to detect an async-await error by looking at the obligation causes, looking
        // for only generators, generator witnesses, opaque types or `std::future::GenFuture` to
        // be present.
        //
        // When a future does not implement a trait because of a captured type in one of the
        // generators somewhere in the call stack, then the result is a chain of obligations.
        // Given a `async fn` A that calls a `async fn` B which captures a non-send type and that
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
        // The first obligations in the chain can be used to get the details of the type that is
        // captured but the entire chain must be inspected to detect this case.
        let mut generator = None;
        let mut next_code = Some(&obligation.cause.code);
        while let Some(code) = next_code {
            debug!("note_obligation_cause_for_async_await: code={:?}", code);
            match code {
                ObligationCauseCode::BuiltinDerivedObligation(derived_obligation) |
                ObligationCauseCode::ImplDerivedObligation(derived_obligation) => {
                    debug!("note_obligation_cause_for_async_await: self_ty.kind={:?}",
                           derived_obligation.parent_trait_ref.self_ty().kind);
                    match derived_obligation.parent_trait_ref.self_ty().kind {
                        ty::Adt(ty::AdtDef { did, .. }, ..) if
                            self.tcx.is_diagnostic_item(sym::gen_future, *did) => {},
                        ty::Generator(did, ..) => generator = generator.or(Some(did)),
                        ty::GeneratorWitness(_) | ty::Opaque(..) => {},
                        _ => return false,
                    }

                    next_code = Some(derived_obligation.parent_code.as_ref());
                },
                ObligationCauseCode::ItemObligation(_) | ObligationCauseCode::BindingObligation(..)
                    if generator.is_some() => break,
                _ => return false,
            }
        }

        let generator_did = generator.expect("can only reach this if there was a generator");

        // Only continue to add a note if the generator is from an `async` function.
        let parent_node = self.tcx.parent(generator_did)
            .and_then(|parent_did| self.tcx.hir().get_if_local(parent_did));
        debug!("note_obligation_cause_for_async_await: parent_node={:?}", parent_node);
        if let Some(hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Fn(sig, _, _),
            ..
        })) = parent_node {
            debug!("note_obligation_cause_for_async_await: header={:?}", sig.header);
            if sig.header.asyncness != hir::IsAsync::Async {
                return false;
            }
        }

        let span = self.tcx.def_span(generator_did);
        let tables = self.tcx.typeck_tables_of(generator_did);
        debug!("note_obligation_cause_for_async_await: generator_did={:?} span={:?} ",
               generator_did, span);

        // Look for a type inside the generator interior that matches the target type to get
        // a span.
        let target_span = tables.generator_interior_types.iter()
            .find(|ty::GeneratorInteriorTypeCause { ty, .. }| ty::TyS::same_type(*ty, target_ty))
            .map(|ty::GeneratorInteriorTypeCause { span, scope_span, .. }|
                 (span, source_map.span_to_snippet(*span), scope_span));
        if let Some((target_span, Ok(snippet), scope_span)) = target_span {
            // Look at the last interior type to get a span for the `.await`.
            let await_span = tables.generator_interior_types.iter().map(|i| i.span).last().unwrap();
            let mut span = MultiSpan::from_span(await_span);
            span.push_span_label(
                await_span, format!("await occurs here, with `{}` maybe used later", snippet));

            span.push_span_label(*target_span, format!("has type `{}`", target_ty));

            // If available, use the scope span to annotate the drop location.
            if let Some(scope_span) = scope_span {
                span.push_span_label(
                    source_map.end_point(*scope_span),
                    format!("`{}` is later dropped here", snippet),
                );
            }

            err.span_note(span, &format!(
                "future does not implement `{}` as this value is used across an await",
                trait_ref,
            ));

            // Add a note for the item obligation that remains - normally a note pointing to the
            // bound that introduced the obligation (e.g. `T: Send`).
            debug!("note_obligation_cause_for_async_await: next_code={:?}", next_code);
            self.note_obligation_cause_code(
                err,
                &obligation.predicate,
                next_code.unwrap(),
                &mut Vec::new(),
            );

            true
        } else {
            false
        }
    }

    fn note_obligation_cause_code<T>(&self,
                                     err: &mut DiagnosticBuilder<'_>,
                                     predicate: &T,
                                     cause_code: &ObligationCauseCode<'tcx>,
                                     obligated_types: &mut Vec<&ty::TyS<'tcx>>)
        where T: fmt::Display
    {
        let tcx = self.tcx;
        match *cause_code {
            ObligationCauseCode::ExprAssignable |
            ObligationCauseCode::MatchExpressionArm { .. } |
            ObligationCauseCode::MatchExpressionArmPattern { .. } |
            ObligationCauseCode::IfExpression { .. } |
            ObligationCauseCode::IfExpressionWithNoElse |
            ObligationCauseCode::MainFunctionType |
            ObligationCauseCode::StartFunctionType |
            ObligationCauseCode::IntrinsicType |
            ObligationCauseCode::MethodReceiver |
            ObligationCauseCode::ReturnNoExpression |
            ObligationCauseCode::MiscObligation => {}
            ObligationCauseCode::SliceOrArrayElem => {
                err.note("slice and array elements must have `Sized` type");
            }
            ObligationCauseCode::TupleElem => {
                err.note("only the last element of a tuple may have a dynamically sized type");
            }
            ObligationCauseCode::ProjectionWf(data) => {
                err.note(&format!(
                    "required so that the projection `{}` is well-formed",
                    data,
                ));
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
                    region,
                    object_ty,
                ));
            }
            ObligationCauseCode::ItemObligation(item_def_id) => {
                let item_name = tcx.def_path_str(item_def_id);
                let msg = format!("required by `{}`", item_name);

                if let Some(sp) = tcx.hir().span_if_local(item_def_id) {
                    let sp = tcx.sess.source_map().def_span(sp);
                    err.span_label(sp, &msg);
                } else {
                    err.note(&msg);
                }
            }
            ObligationCauseCode::BindingObligation(item_def_id, span) => {
                let item_name = tcx.def_path_str(item_def_id);
                let msg = format!("required by this bound in `{}`", item_name);
                if let Some(ident) = tcx.opt_item_name(item_def_id) {
                    err.span_label(ident.span, "");
                }
                if span != DUMMY_SP {
                    err.span_label(span, &msg);
                } else {
                    err.note(&msg);
                }
            }
            ObligationCauseCode::ObjectCastObligation(object_ty) => {
                err.note(&format!("required for the cast to the object type `{}`",
                                  self.ty_to_string(object_ty)));
            }
            ObligationCauseCode::Coercion { source: _, target } => {
                err.note(&format!("required by cast to type `{}`",
                                  self.ty_to_string(target)));
            }
            ObligationCauseCode::RepeatVec(suggest_const_in_array_repeat_expressions) => {
                err.note("the `Copy` trait is required because the \
                          repeated element will be copied");
                if suggest_const_in_array_repeat_expressions {
                    err.note("this array initializer can be evaluated at compile-time, for more \
                              information, see issue \
                              https://github.com/rust-lang/rust/issues/49147");
                    if tcx.sess.opts.unstable_features.is_nightly_build() {
                        err.help("add `#![feature(const_in_array_repeat_expressions)]` to the \
                                  crate attributes to enable");
                    }
                }
            }
            ObligationCauseCode::VariableType(_) => {
                err.note("all local variables must have a statically known size");
                if !self.tcx.features().unsized_locals {
                    err.help("unsized locals are gated as an unstable feature");
                }
            }
            ObligationCauseCode::SizedArgumentType => {
                err.note("all function arguments must have a statically known size");
                if !self.tcx.features().unsized_locals {
                    err.help("unsized locals are gated as an unstable feature");
                }
            }
            ObligationCauseCode::SizedReturnType => {
                err.note("the return type of a function must have a \
                          statically known size");
            }
            ObligationCauseCode::SizedYieldType => {
                err.note("the yield type of a generator must have a \
                          statically known size");
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
            ObligationCauseCode::FieldSized { adt_kind: ref item, last } => {
                match *item {
                    AdtKind::Struct => {
                        if last {
                            err.note("the last field of a packed struct may only have a \
                                      dynamically sized type if it does not need drop to be run");
                        } else {
                            err.note("only the last field of a struct may have a dynamically \
                                      sized type");
                        }
                    }
                    AdtKind::Union => {
                        err.note("no field of a union may have a dynamically sized type");
                    }
                    AdtKind::Enum => {
                        err.note("no field of an enum variant may have a dynamically sized type");
                    }
                }
            }
            ObligationCauseCode::ConstSized => {
                err.note("constant expressions must have a statically known size");
            }
            ObligationCauseCode::ConstPatternStructural => {
                err.note("constants used for pattern-matching must derive `PartialEq` and `Eq`");
            }
            ObligationCauseCode::SharedStatic => {
                err.note("shared static variables must have a type that implements `Sync`");
            }
            ObligationCauseCode::BuiltinDerivedObligation(ref data) => {
                let parent_trait_ref = self.resolve_vars_if_possible(&data.parent_trait_ref);
                let ty = parent_trait_ref.skip_binder().self_ty();
                err.note(&format!("required because it appears within the type `{}`", ty));
                obligated_types.push(ty);

                let parent_predicate = parent_trait_ref.to_predicate();
                if !self.is_recursive_obligation(obligated_types, &data.parent_code) {
                    self.note_obligation_cause_code(err,
                                                    &parent_predicate,
                                                    &data.parent_code,
                                                    obligated_types);
                }
            }
            ObligationCauseCode::ImplDerivedObligation(ref data) => {
                let parent_trait_ref = self.resolve_vars_if_possible(&data.parent_trait_ref);
                err.note(
                    &format!("required because of the requirements on the impl of `{}` for `{}`",
                             parent_trait_ref,
                             parent_trait_ref.skip_binder().self_ty()));
                let parent_predicate = parent_trait_ref.to_predicate();
                self.note_obligation_cause_code(err,
                                                &parent_predicate,
                                                &data.parent_code,
                                                obligated_types);
            }
            ObligationCauseCode::CompareImplMethodObligation { .. } => {
                err.note(
                    &format!("the requirement `{}` appears on the impl method \
                              but not on the corresponding trait method",
                             predicate));
            }
            ObligationCauseCode::ReturnType |
            ObligationCauseCode::ReturnValue(_) |
            ObligationCauseCode::BlockTailExpression(_) => (),
            ObligationCauseCode::TrivialBound => {
                err.help("see issue #48214");
                if tcx.sess.opts.unstable_features.is_nightly_build() {
                    err.help("add `#![feature(trivial_bounds)]` to the \
                              crate attributes to enable",
                    );
                }
            }
            ObligationCauseCode::AssocTypeBound(ref data) => {
                err.span_label(data.original, "associated type defined here");
                if let Some(sp) = data.impl_span {
                    err.span_label(sp, "in this `impl` item");
                }
                for sp in &data.bounds {
                    err.span_label(*sp, "restricted in this bound");
                }
            }
        }
    }

    fn suggest_new_overflow_limit(&self, err: &mut DiagnosticBuilder<'_>) {
        let current_limit = self.tcx.sess.recursion_limit.get();
        let suggested_limit = current_limit * 2;
        err.help(&format!("consider adding a `#![recursion_limit=\"{}\"]` attribute to your crate",
                          suggested_limit));
    }

    fn is_recursive_obligation(&self,
                               obligated_types: &mut Vec<&ty::TyS<'tcx>>,
                               cause_code: &ObligationCauseCode<'tcx>) -> bool {
        if let ObligationCauseCode::BuiltinDerivedObligation(ref data) = cause_code {
            let parent_trait_ref = self.resolve_vars_if_possible(&data.parent_trait_ref);

            if obligated_types.iter().any(|ot| ot == &parent_trait_ref.skip_binder().self_ty()) {
                return true;
            }
        }
        false
    }
}

/// Summarizes information
#[derive(Clone)]
pub enum ArgKind {
    /// An argument of non-tuple type. Parameters are (name, ty)
    Arg(String, String),

    /// An argument of tuple type. For a "found" argument, the span is
    /// the locationo in the source of the pattern. For a "expected"
    /// argument, it will be None. The vector is a list of (name, ty)
    /// strings for the components of the tuple.
    Tuple(Option<Span>, Vec<(String, String)>),
}

impl ArgKind {
    fn empty() -> ArgKind {
        ArgKind::Arg("_".to_owned(), "_".to_owned())
    }

    /// Creates an `ArgKind` from the expected type of an
    /// argument. It has no name (`_`) and an optional source span.
    pub fn from_expected_ty(t: Ty<'_>, span: Option<Span>) -> ArgKind {
        match t.kind {
            ty::Tuple(ref tys) => ArgKind::Tuple(
                span,
                tys.iter()
                   .map(|ty| ("_".to_owned(), ty.to_string()))
                   .collect::<Vec<_>>()
            ),
            _ => ArgKind::Arg("_".to_owned(), t.to_string()),
        }
    }
}
