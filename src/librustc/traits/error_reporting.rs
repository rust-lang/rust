// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{
    FulfillmentError,
    FulfillmentErrorCode,
    MismatchedProjectionTypes,
    Obligation,
    ObligationCause,
    ObligationCauseCode,
    OutputTypeParameterMismatch,
    TraitNotObjectSafe,
    PredicateObligation,
    SelectionContext,
    SelectionError,
    ObjectSafetyViolation,
};

use errors::DiagnosticBuilder;
use fmt_macros::{Parser, Piece, Position};
use hir::{self, intravisit, Local, Pat, Body};
use hir::intravisit::{Visitor, NestedVisitorMap};
use hir::map::NodeExpr;
use hir::def_id::DefId;
use infer::{self, InferCtxt};
use infer::type_variable::TypeVariableOrigin;
use rustc::lint::builtin::EXTRA_REQUIREMENT_IN_IMPL;
use std::fmt;
use syntax::ast::{self, NodeId};
use ty::{self, AdtKind, ToPredicate, ToPolyTraitRef, Ty, TyCtxt, TypeFoldable, TyInfer, TyVar};
use ty::error::{ExpectedFound, TypeError};
use ty::fast_reject;
use ty::fold::TypeFolder;
use ty::subst::Subst;
use ty::SubtypePredicate;
use util::nodemap::{FxHashMap, FxHashSet};

use syntax_pos::{DUMMY_SP, Span};


#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TraitErrorKey<'tcx> {
    span: Span,
    predicate: ty::Predicate<'tcx>
}

impl<'a, 'gcx, 'tcx> TraitErrorKey<'tcx> {
    fn from_error(infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                  e: &FulfillmentError<'tcx>) -> Self {
        let predicate =
            infcx.resolve_type_vars_if_possible(&e.obligation.predicate);
        TraitErrorKey {
            span: e.obligation.cause.span,
            predicate: infcx.tcx.erase_regions(&predicate)
        }
    }
}

struct FindLocalByTypeVisitor<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    target_ty: &'a Ty<'tcx>,
    hir_map: &'a hir::map::Map<'gcx>,
    found_local_pattern: Option<&'gcx Pat>,
    found_arg_pattern: Option<&'gcx Pat>,
}

impl<'a, 'gcx, 'tcx> FindLocalByTypeVisitor<'a, 'gcx, 'tcx> {
    fn node_matches_type(&mut self, node_id: &'gcx NodeId) -> bool {
        match self.infcx.tables.borrow().node_types.get(node_id) {
            Some(&ty) => {
                let ty = self.infcx.resolve_type_vars_if_possible(&ty);
                ty.walk().any(|inner_ty| {
                    inner_ty == *self.target_ty || match (&inner_ty.sty, &self.target_ty.sty) {
                        (&TyInfer(TyVar(a_vid)), &TyInfer(TyVar(b_vid))) => {
                            self.infcx
                                .type_variables
                                .borrow_mut()
                                .sub_unified(a_vid, b_vid)
                        }
                        _ => false,
                    }
                })
            }
            _ => false,
        }
    }
}

impl<'a, 'gcx, 'tcx> Visitor<'gcx> for FindLocalByTypeVisitor<'a, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'gcx> {
        NestedVisitorMap::OnlyBodies(&self.hir_map)
    }

    fn visit_local(&mut self, local: &'gcx Local) {
        if self.found_local_pattern.is_none() && self.node_matches_type(&local.id) {
            self.found_local_pattern = Some(&*local.pat);
        }
        intravisit::walk_local(self, local);
    }

    fn visit_body(&mut self, body: &'gcx Body) {
        for argument in &body.arguments {
            if self.found_arg_pattern.is_none() && self.node_matches_type(&argument.id) {
                self.found_arg_pattern = Some(&*argument.pat);
            }
        }
        intravisit::walk_body(self, body);
    }
}

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    pub fn report_fulfillment_errors(&self, errors: &Vec<FulfillmentError<'tcx>>) {
        for error in errors {
            self.report_fulfillment_error(error);
        }
    }

    fn report_fulfillment_error(&self,
                                error: &FulfillmentError<'tcx>) {
        let error_key = TraitErrorKey::from_error(self, error);
        debug!("report_fulfillment_errors({:?}) - key={:?}",
               error, error_key);
        if !self.reported_trait_errors.borrow_mut().insert(error_key) {
            debug!("report_fulfillment_errors: skipping duplicate");
            return;
        }
        match error.code {
            FulfillmentErrorCode::CodeSelectionError(ref e) => {
                self.report_selection_error(&error.obligation, e);
            }
            FulfillmentErrorCode::CodeProjectionError(ref e) => {
                self.report_projection_error(&error.obligation, e);
            }
            FulfillmentErrorCode::CodeAmbiguity => {
                self.maybe_report_ambiguity(&error.obligation);
            }
            FulfillmentErrorCode::CodeSubtypeError(ref expected_found, ref err) => {
                self.report_mismatched_types(&error.obligation.cause,
                                             expected_found.expected,
                                             expected_found.found,
                                             err.clone())
                    .emit();
            }
        }
    }

    fn report_projection_error(&self,
                               obligation: &PredicateObligation<'tcx>,
                               error: &MismatchedProjectionTypes<'tcx>)
    {
        let predicate =
            self.resolve_type_vars_if_possible(&obligation.predicate);

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
                let (data, _) = self.replace_late_bound_regions_with_fresh_var(
                    obligation.cause.span,
                    infer::LateBoundRegionConversionTime::HigherRankedType,
                    data);
                let normalized = super::normalize_projection_type(
                    &mut selcx,
                    data.projection_ty,
                    obligation.cause.clone(),
                    0
                );
                if let Err(error) = self.eq_types(
                    false, &obligation.cause,
                    data.ty, normalized.value
                ) {
                    values = Some(infer::ValuePairs::Types(ExpectedFound {
                        expected: normalized.value,
                        found: data.ty,
                    }));
                    err_buf = error;
                    err = &err_buf;
                }
            }

            let mut diag = struct_span_err!(
                self.tcx.sess, obligation.cause.span, E0271,
                "type mismatch resolving `{}`", predicate
            );
            self.note_type_err(&mut diag, &obligation.cause, None, values, err);
            self.note_obligation_cause(&mut diag, obligation);
            diag.emit();
        });
    }

    fn fuzzy_match_tys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
        /// returns the fuzzy category of a given type, or None
        /// if the type can be equated to any type.
        fn type_category<'tcx>(t: Ty<'tcx>) -> Option<u32> {
            match t.sty {
                ty::TyBool => Some(0),
                ty::TyChar => Some(1),
                ty::TyStr => Some(2),
                ty::TyInt(..) | ty::TyUint(..) | ty::TyInfer(ty::IntVar(..)) => Some(3),
                ty::TyFloat(..) | ty::TyInfer(ty::FloatVar(..)) => Some(4),
                ty::TyRef(..) | ty::TyRawPtr(..) => Some(5),
                ty::TyArray(..) | ty::TySlice(..) => Some(6),
                ty::TyFnDef(..) | ty::TyFnPtr(..) => Some(7),
                ty::TyDynamic(..) => Some(8),
                ty::TyClosure(..) => Some(9),
                ty::TyTuple(..) => Some(10),
                ty::TyProjection(..) => Some(11),
                ty::TyParam(..) => Some(12),
                ty::TyAnon(..) => Some(13),
                ty::TyNever => Some(14),
                ty::TyAdt(adt, ..) => match adt.adt_kind() {
                    AdtKind::Struct => Some(15),
                    AdtKind::Union => Some(16),
                    AdtKind::Enum => Some(17),
                },
                ty::TyInfer(..) | ty::TyError => None
            }
        }

        match (type_category(a), type_category(b)) {
            (Some(cat_a), Some(cat_b)) => match (&a.sty, &b.sty) {
                (&ty::TyAdt(def_a, _), &ty::TyAdt(def_b, _)) => def_a == def_b,
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

        let trait_ref = tcx.erase_late_bound_regions(&trait_ref);
        let trait_self_ty = trait_ref.self_ty();

        let mut self_match_impls = vec![];
        let mut fuzzy_match_impls = vec![];

        self.tcx.trait_def(trait_ref.def_id)
            .for_each_relevant_impl(self.tcx, trait_self_ty, |def_id| {
                let impl_substs = self.fresh_substs_for_item(obligation.cause.span, def_id);
                let impl_trait_ref = tcx
                    .impl_trait_ref(def_id)
                    .unwrap()
                    .subst(tcx, impl_substs);

                let impl_self_ty = impl_trait_ref.self_ty();

                if let Ok(..) = self.can_equate(&trait_self_ty, &impl_self_ty) {
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

        if tcx.has_attr(impl_def_id, "rustc_on_unimplemented") {
            Some(impl_def_id)
        } else {
            None
        }
    }

    fn on_unimplemented_note(&self,
                             trait_ref: ty::PolyTraitRef<'tcx>,
                             obligation: &PredicateObligation<'tcx>) -> Option<String> {
        let def_id = self.impl_similar_to(trait_ref, obligation)
            .unwrap_or(trait_ref.def_id());
        let trait_ref = trait_ref.skip_binder();

        let span = obligation.cause.span;
        let mut report = None;
        if let Some(item) = self.tcx
            .get_attrs(def_id)
            .into_iter()
            .filter(|a| a.check_name("rustc_on_unimplemented"))
            .next()
        {
            let err_sp = item.span.substitute_dummy(span);
            let trait_str = self.tcx.item_path_str(trait_ref.def_id);
            if let Some(istring) = item.value_str() {
                let istring = &*istring.as_str();
                let generics = self.tcx.generics_of(trait_ref.def_id);
                let generic_map = generics.types.iter().map(|param| {
                    (param.name.as_str().to_string(),
                        trait_ref.substs.type_for_def(param).to_string())
                }).collect::<FxHashMap<String, String>>();
                let parser = Parser::new(istring);
                let mut errored = false;
                let err: String = parser.filter_map(|p| {
                    match p {
                        Piece::String(s) => Some(s),
                        Piece::NextArgument(a) => match a.position {
                            Position::ArgumentNamed(s) => match generic_map.get(s) {
                                Some(val) => Some(val),
                                None => {
                                    span_err!(self.tcx.sess, err_sp, E0272,
                                              "the #[rustc_on_unimplemented] attribute on trait \
                                               definition for {} refers to non-existent type \
                                               parameter {}",
                                              trait_str, s);
                                    errored = true;
                                    None
                                }
                            },
                            _ => {
                                span_err!(self.tcx.sess, err_sp, E0273,
                                          "the #[rustc_on_unimplemented] attribute on trait \
                                           definition for {} must have named format arguments, eg \
                                           `#[rustc_on_unimplemented = \"foo {{T}}\"]`",
                                          trait_str);
                                errored = true;
                                None
                            }
                        }
                    }
                }).collect();
                // Report only if the format string checks out
                if !errored {
                    report = Some(err);
                }
            } else {
                span_err!(self.tcx.sess, err_sp, E0274,
                                        "the #[rustc_on_unimplemented] attribute on \
                                                    trait definition for {} must have a value, \
                                                    eg `#[rustc_on_unimplemented = \"foo\"]`",
                                                    trait_str);
            }
        }
        report
    }

    fn find_similar_impl_candidates(&self,
                                    trait_ref: ty::PolyTraitRef<'tcx>)
                                    -> Vec<ty::TraitRef<'tcx>>
    {
        let simp = fast_reject::simplify_type(self.tcx,
                                              trait_ref.skip_binder().self_ty(),
                                              true);
        let mut impl_candidates = Vec::new();
        let trait_def = self.tcx.trait_def(trait_ref.def_id());

        match simp {
            Some(simp) => trait_def.for_each_impl(self.tcx, |def_id| {
                let imp = self.tcx.impl_trait_ref(def_id).unwrap();
                let imp_simp = fast_reject::simplify_type(self.tcx,
                                                          imp.self_ty(),
                                                          true);
                if let Some(imp_simp) = imp_simp {
                    if simp != imp_simp {
                        return;
                    }
                }
                impl_candidates.push(imp);
            }),
            None => trait_def.for_each_impl(self.tcx, |def_id| {
                impl_candidates.push(
                    self.tcx.impl_trait_ref(def_id).unwrap());
            })
        };
        impl_candidates
    }

    fn report_similar_impl_candidates(&self,
                                      impl_candidates: Vec<ty::TraitRef<'tcx>>,
                                      err: &mut DiagnosticBuilder)
    {
        if impl_candidates.is_empty() {
            return;
        }

        let end = if impl_candidates.len() <= 5 {
            impl_candidates.len()
        } else {
            4
        };
        err.help(&format!("the following implementations were found:{}{}",
                          &impl_candidates[0..end].iter().map(|candidate| {
                              format!("\n  {:?}", candidate)
                          }).collect::<String>(),
                          if impl_candidates.len() > 5 {
                              format!("\nand {} others", impl_candidates.len() - 4)
                          } else {
                              "".to_owned()
                          }
                          ));
    }

    /// Reports that an overflow has occurred and halts compilation. We
    /// halt compilation unconditionally because it is important that
    /// overflows never be masked -- they basically represent computations
    /// whose result could not be truly determined and thus we can't say
    /// if the program type checks or not -- and they are unusual
    /// occurrences in any case.
    pub fn report_overflow_error<T>(&self,
                                    obligation: &Obligation<'tcx, T>,
                                    suggest_increasing_limit: bool) -> !
        where T: fmt::Display + TypeFoldable<'tcx>
    {
        let predicate =
            self.resolve_type_vars_if_possible(&obligation.predicate);
        let mut err = struct_span_err!(self.tcx.sess, obligation.cause.span, E0275,
                                       "overflow evaluating the requirement `{}`",
                                       predicate);

        if suggest_increasing_limit {
            self.suggest_new_overflow_limit(&mut err);
        }

        self.note_obligation_cause(&mut err, obligation);

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
        let cycle = self.resolve_type_vars_if_possible(&cycle.to_owned());
        assert!(cycle.len() > 0);

        debug!("report_overflow_error_cycle: cycle={:?}", cycle);

        self.report_overflow_error(&cycle[0], false);
    }

    pub fn report_extra_impl_obligation(&self,
                                        error_span: Span,
                                        item_name: ast::Name,
                                        _impl_item_def_id: DefId,
                                        trait_item_def_id: DefId,
                                        requirement: &fmt::Display,
                                        lint_id: Option<ast::NodeId>) // (*)
                                        -> DiagnosticBuilder<'tcx>
    {
        // (*) This parameter is temporary and used only for phasing
        // in the bug fix to #18937. If it is `Some`, it has a kind of
        // weird effect -- the diagnostic is reported as a lint, and
        // the builder which is returned is marked as canceled.

        let mut err =
            struct_span_err!(self.tcx.sess,
                             error_span,
                             E0276,
                             "impl has stricter requirements than trait");

        if let Some(trait_item_span) = self.tcx.hir.span_if_local(trait_item_def_id) {
            let span = self.tcx.sess.codemap().def_span(trait_item_span);
            err.span_label(span, &format!("definition of `{}` from trait", item_name));
        }

        err.span_label(
            error_span,
            &format!("impl has extra requirement {}", requirement));

        if let Some(node_id) = lint_id {
            self.tcx.sess.add_lint_diagnostic(EXTRA_REQUIREMENT_IN_IMPL,
                                              node_id,
                                              (*err).clone());
            err.cancel();
        }

        err
    }


    /// Get the parent trait chain start
    fn get_parent_trait_ref(&self, code: &ObligationCauseCode<'tcx>) -> Option<String> {
        match code {
            &ObligationCauseCode::BuiltinDerivedObligation(ref data) => {
                let parent_trait_ref = self.resolve_type_vars_if_possible(
                    &data.parent_trait_ref);
                match self.get_parent_trait_ref(&data.parent_code) {
                    Some(t) => Some(t),
                    None => Some(format!("{}", parent_trait_ref.0.self_ty())),
                }
            }
            _ => None,
        }
    }

    pub fn report_selection_error(&self,
                                  obligation: &PredicateObligation<'tcx>,
                                  error: &SelectionError<'tcx>)
    {
        let span = obligation.cause.span;

        let mut err = match *error {
            SelectionError::Unimplemented => {
                if let ObligationCauseCode::CompareImplMethodObligation {
                    item_name, impl_item_def_id, trait_item_def_id, lint_id
                } = obligation.cause.code {
                    self.report_extra_impl_obligation(
                        span,
                        item_name,
                        impl_item_def_id,
                        trait_item_def_id,
                        &format!("`{}`", obligation.predicate),
                        lint_id)
                        .emit();
                    return;
                }
                match obligation.predicate {
                    ty::Predicate::Trait(ref trait_predicate) => {
                        let trait_predicate =
                            self.resolve_type_vars_if_possible(trait_predicate);

                        if self.tcx.sess.has_errors() && trait_predicate.references_error() {
                            return;
                        }
                        let trait_ref = trait_predicate.to_poly_trait_ref();
                        let (post_message, pre_message) =
                            self.get_parent_trait_ref(&obligation.cause.code)
                                .map(|t| (format!(" in `{}`", t), format!("within `{}`, ", t)))
                                .unwrap_or((String::new(), String::new()));
                        let mut err = struct_span_err!(
                            self.tcx.sess,
                            span,
                            E0277,
                            "the trait bound `{}` is not satisfied{}",
                            trait_ref.to_predicate(),
                            post_message);

                        // Try to report a help message
                        if !trait_ref.has_infer_types() &&
                            self.predicate_can_apply(trait_ref) {
                            // If a where-clause may be useful, remind the
                            // user that they can add it.
                            //
                            // don't display an on-unimplemented note, as
                            // these notes will often be of the form
                            //     "the type `T` can't be frobnicated"
                            // which is somewhat confusing.
                            err.help(&format!("consider adding a `where {}` bound",
                                                trait_ref.to_predicate()));
                        } else if let Some(s) = self.on_unimplemented_note(trait_ref, obligation) {
                            // If it has a custom "#[rustc_on_unimplemented]"
                            // error message, let's display it!
                            err.note(&s);
                        } else {
                            // Can't show anything else useful, try to find similar impls.
                            let impl_candidates = self.find_similar_impl_candidates(trait_ref);
                            self.report_similar_impl_candidates(impl_candidates, &mut err);
                        }

                        err.span_label(span,
                                       &format!("{}the trait `{}` is not implemented for `{}`",
                                                pre_message,
                                                trait_ref,
                                                trait_ref.self_ty()));
                        err
                    }

                    ty::Predicate::Subtype(ref predicate) => {
                        // Errors for Subtype predicates show up as
                        // `FulfillmentErrorCode::CodeSubtypeError`,
                        // not selection error.
                        span_bug!(span, "subtype requirement gave wrong error: `{:?}`", predicate)
                    }

                    ty::Predicate::Equate(ref predicate) => {
                        let predicate = self.resolve_type_vars_if_possible(predicate);
                        let err = self.equality_predicate(&obligation.cause,
                                                            &predicate).err().unwrap();
                        struct_span_err!(self.tcx.sess, span, E0278,
                            "the requirement `{}` is not satisfied (`{}`)",
                            predicate, err)
                    }

                    ty::Predicate::RegionOutlives(ref predicate) => {
                        let predicate = self.resolve_type_vars_if_possible(predicate);
                        let err = self.region_outlives_predicate(&obligation.cause,
                                                                    &predicate).err().unwrap();
                        struct_span_err!(self.tcx.sess, span, E0279,
                            "the requirement `{}` is not satisfied (`{}`)",
                            predicate, err)
                    }

                    ty::Predicate::Projection(..) | ty::Predicate::TypeOutlives(..) => {
                        let predicate =
                            self.resolve_type_vars_if_possible(&obligation.predicate);
                        struct_span_err!(self.tcx.sess, span, E0280,
                            "the requirement `{}` is not satisfied",
                            predicate)
                    }

                    ty::Predicate::ObjectSafe(trait_def_id) => {
                        let violations = self.tcx.object_safety_violations(trait_def_id);
                        self.tcx.report_object_safety_error(span,
                                                            trait_def_id,
                                                            violations)
                    }

                    ty::Predicate::ClosureKind(closure_def_id, kind) => {
                        let found_kind = self.closure_kind(closure_def_id).unwrap();
                        let closure_span = self.tcx.hir.span_if_local(closure_def_id).unwrap();
                        let mut err = struct_span_err!(
                            self.tcx.sess, closure_span, E0525,
                            "expected a closure that implements the `{}` trait, \
                                but this closure only implements `{}`",
                            kind,
                            found_kind);
                        err.span_note(
                            obligation.cause.span,
                            &format!("the requirement to implement \
                                        `{}` derives from here", kind));
                        err.emit();
                        return;
                    }

                    ty::Predicate::WellFormed(ty) => {
                        // WF predicates cannot themselves make
                        // errors. They can only block due to
                        // ambiguity; otherwise, they always
                        // degenerate into other obligations
                        // (which may fail).
                        span_bug!(span, "WF predicate not satisfied for {:?}", ty);
                    }
                }
            }

            OutputTypeParameterMismatch(ref expected_trait_ref, ref actual_trait_ref, ref e) => {
                let expected_trait_ref = self.resolve_type_vars_if_possible(&*expected_trait_ref);
                let actual_trait_ref = self.resolve_type_vars_if_possible(&*actual_trait_ref);
                if actual_trait_ref.self_ty().references_error() {
                    return;
                }
                let expected_trait_ty = expected_trait_ref.self_ty();
                let found_span = expected_trait_ty.ty_to_def_id().and_then(|did| {
                    self.tcx.hir.span_if_local(did)
                });

                if let &TypeError::TupleSize(ref expected_found) = e {
                    // Expected `|x| { }`, found `|x, y| { }`
                    self.report_arg_count_mismatch(span,
                                                   found_span,
                                                   expected_found.expected,
                                                   expected_found.found,
                                                   expected_trait_ty.is_closure())
                } else if let &TypeError::Sorts(ref expected_found) = e {
                    let expected = if let ty::TyTuple(tys, _) = expected_found.expected.sty {
                        tys.len()
                    } else {
                        1
                    };
                    let found = if let ty::TyTuple(tys, _) = expected_found.found.sty {
                        tys.len()
                    } else {
                        1
                    };

                    if expected != found {
                        // Expected `|| { }`, found `|x, y| { }`
                        // Expected `fn(x) -> ()`, found `|| { }`
                        self.report_arg_count_mismatch(span,
                                                       found_span,
                                                       expected,
                                                       found,
                                                       expected_trait_ty.is_closure())
                    } else {
                        self.report_type_argument_mismatch(span,
                                                            found_span,
                                                            expected_trait_ty,
                                                            expected_trait_ref,
                                                            actual_trait_ref,
                                                            e)
                    }
                } else {
                    self.report_type_argument_mismatch(span,
                                                        found_span,
                                                        expected_trait_ty,
                                                        expected_trait_ref,
                                                        actual_trait_ref,
                                                        e)
                }
            }

            TraitNotObjectSafe(did) => {
                let violations = self.tcx.object_safety_violations(did);
                self.tcx.report_object_safety_error(span, did,
                                                    violations)
            }
        };
        self.note_obligation_cause(&mut err, obligation);
        err.emit();
    }

    fn report_type_argument_mismatch(&self,
                                      span: Span,
                                      found_span: Option<Span>,
                                      expected_ty: Ty<'tcx>,
                                      expected_ref: ty::PolyTraitRef<'tcx>,
                                      found_ref: ty::PolyTraitRef<'tcx>,
                                      type_error: &TypeError<'tcx>)
        -> DiagnosticBuilder<'tcx>
    {
        let mut err = struct_span_err!(self.tcx.sess, span, E0281,
            "type mismatch: `{}` implements the trait `{}`, but the trait `{}` is required",
            expected_ty,
            expected_ref,
            found_ref);

        err.span_label(span, &format!("{}", type_error));

        if let Some(sp) = found_span {
            err.span_label(span, &format!("requires `{}`", found_ref));
            err.span_label(sp, &format!("implements `{}`", expected_ref));
        }

        err
    }

    fn report_arg_count_mismatch(&self,
                                 span: Span,
                                 found_span: Option<Span>,
                                 expected: usize,
                                 found: usize,
                                 is_closure: bool)
        -> DiagnosticBuilder<'tcx>
    {
        let mut err = struct_span_err!(self.tcx.sess, span, E0593,
            "{} takes {} argument{} but {} argument{} {} required",
            if is_closure { "closure" } else { "function" },
            found,
            if found == 1 { "" } else { "s" },
            expected,
            if expected == 1 { "" } else { "s" },
            if expected == 1 { "is" } else { "are" });

        err.span_label(span, &format!("expected {} that takes {} argument{}",
                                      if is_closure { "closure" } else { "function" },
                                      expected,
                                      if expected == 1 { "" } else { "s" }));
        if let Some(span) = found_span {
            err.span_label(span, &format!("takes {} argument{}",
                                          found,
                                          if found == 1 { "" } else { "s" }));
        }
        err
    }
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn recursive_type_with_infinite_size_error(self,
                                                   type_def_id: DefId)
                                                   -> DiagnosticBuilder<'tcx>
    {
        assert!(type_def_id.is_local());
        let span = self.hir.span_if_local(type_def_id).unwrap();
        let span = self.sess.codemap().def_span(span);
        let mut err = struct_span_err!(self.sess, span, E0072,
                                       "recursive type `{}` has infinite size",
                                       self.item_path_str(type_def_id));
        err.span_label(span, &format!("recursive type has infinite size"));
        err.help(&format!("insert indirection (e.g., a `Box`, `Rc`, or `&`) \
                           at some point to make `{}` representable",
                          self.item_path_str(type_def_id)));
        err
    }

    pub fn report_object_safety_error(self,
                                      span: Span,
                                      trait_def_id: DefId,
                                      violations: Vec<ObjectSafetyViolation>)
                                      -> DiagnosticBuilder<'tcx>
    {
        let trait_str = self.item_path_str(trait_def_id);
        let span = self.sess.codemap().def_span(span);
        let mut err = struct_span_err!(
            self.sess, span, E0038,
            "the trait `{}` cannot be made into an object",
            trait_str);
        err.span_label(span, &format!("the trait `{}` cannot be made into an object", trait_str));

        let mut reported_violations = FxHashSet();
        for violation in violations {
            if !reported_violations.insert(violation.clone()) {
                continue;
            }
            err.note(&violation.error_msg());
        }
        err
    }
}

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    fn maybe_report_ambiguity(&self, obligation: &PredicateObligation<'tcx>) {
        // Unable to successfully determine, probably means
        // insufficient type information, but could mean
        // ambiguous impls. The latter *ought* to be a
        // coherence violation, so we don't report it here.

        let predicate = self.resolve_type_vars_if_possible(&obligation.predicate);
        let body_id = hir::BodyId { node_id: obligation.cause.body_id };
        let span = obligation.cause.span;

        debug!("maybe_report_ambiguity(predicate={:?}, obligation={:?})",
               predicate,
               obligation);

        // Ambiguity errors are often caused as fallout from earlier
        // errors. So just ignore them if this infcx is tainted.
        if self.is_tainted_by_errors() {
            return;
        }

        match predicate {
            ty::Predicate::Trait(ref data) => {
                let trait_ref = data.to_poly_trait_ref();
                let self_ty = trait_ref.self_ty();
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
                // inhabited. But in any case I just threw in this check for
                // has_errors() to be sure that compilation isn't happening
                // anyway. In that case, why inundate the user.
                if !self.tcx.sess.has_errors() {
                    if
                        self.tcx.lang_items.sized_trait()
                        .map_or(false, |sized_id| sized_id == trait_ref.def_id())
                    {
                        self.need_type_info(body_id, span, self_ty);
                    } else {
                        let mut err = struct_span_err!(self.tcx.sess,
                                                        span, E0283,
                                                        "type annotations required: \
                                                        cannot resolve `{}`",
                                                        predicate);
                        self.note_obligation_cause(&mut err, obligation);
                        err.emit();
                    }
                }
            }

            ty::Predicate::WellFormed(ty) => {
                // Same hacky approach as above to avoid deluging user
                // with error messages.
                if !ty.references_error() && !self.tcx.sess.has_errors() {
                    self.need_type_info(body_id, span, ty);
                }
            }

            ty::Predicate::Subtype(ref data) => {
                if data.references_error() || self.tcx.sess.has_errors() {
                    // no need to overload user in such cases
                } else {
                    let &SubtypePredicate { a_is_expected: _, a, b } = data.skip_binder();
                    // both must be type variables, or the other would've been instantiated
                    assert!(a.is_ty_var() && b.is_ty_var());
                    self.need_type_info(hir::BodyId { node_id: obligation.cause.body_id },
                                        obligation.cause.span,
                                        a);
                }
            }

            _ => {
                if !self.tcx.sess.has_errors() {
                    let mut err = struct_span_err!(self.tcx.sess,
                                                   obligation.cause.span, E0284,
                                                   "type annotations required: \
                                                    cannot resolve `{}`",
                                                   predicate);
                    self.note_obligation_cause(&mut err, obligation);
                    err.emit();
                }
            }
        }
    }

    /// Returns whether the trait predicate may apply for *some* assignment
    /// to the type parameters.
    fn predicate_can_apply(&self, pred: ty::PolyTraitRef<'tcx>) -> bool {
        struct ParamToVarFolder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
            infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
            var_map: FxHashMap<Ty<'tcx>, Ty<'tcx>>
        }

        impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for ParamToVarFolder<'a, 'gcx, 'tcx> {
            fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.infcx.tcx }

            fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
                if let ty::TyParam(ty::ParamTy {name, ..}) = ty.sty {
                    let infcx = self.infcx;
                    self.var_map.entry(ty).or_insert_with(||
                        infcx.next_ty_var(
                            TypeVariableOrigin::TypeParameterDefinition(DUMMY_SP, name)))
                } else {
                    ty.super_fold_with(self)
                }
            }
        }

        self.probe(|_| {
            let mut selcx = SelectionContext::new(self);

            let cleaned_pred = pred.fold_with(&mut ParamToVarFolder {
                infcx: self,
                var_map: FxHashMap()
            });

            let cleaned_pred = super::project::normalize(
                &mut selcx,
                ObligationCause::dummy(),
                &cleaned_pred
            ).value;

            let obligation = Obligation::new(
                ObligationCause::dummy(),
                cleaned_pred.to_predicate()
            );

            selcx.evaluate_obligation(&obligation)
        })
    }

    fn extract_type_name(&self, ty: &'a Ty<'tcx>) -> String {
        if let ty::TyInfer(ty::TyVar(ty_vid)) = (*ty).sty {
            let ty_vars = self.type_variables.borrow();
            if let TypeVariableOrigin::TypeParameterDefinition(_, name) =
                *ty_vars.var_origin(ty_vid) {
                name.to_string()
            } else {
                ty.to_string()
            }
        } else {
            ty.to_string()
        }
    }

    pub fn need_type_info(&self, body_id: hir::BodyId, span: Span, ty: Ty<'tcx>) {
        let ty = self.resolve_type_vars_if_possible(&ty);
        let name = self.extract_type_name(&ty);

        let mut err_span = span;
        let mut labels = vec![(span, format!("cannot infer type for `{}`", name))];

        let mut local_visitor = FindLocalByTypeVisitor {
            infcx: &self,
            target_ty: &ty,
            hir_map: &self.tcx.hir,
            found_local_pattern: None,
            found_arg_pattern: None,
        };

        // #40294: cause.body_id can also be a fn declaration.
        // Currently, if it's anything other than NodeExpr, we just ignore it
        match self.tcx.hir.find(body_id.node_id) {
            Some(NodeExpr(expr)) => local_visitor.visit_expr(expr),
            _ => ()
        }

        if let Some(pattern) = local_visitor.found_arg_pattern {
            err_span = pattern.span;
            // We don't want to show the default label for closures.
            //
            // So, before clearing, the output would look something like this:
            // ```
            // let x = |_| {  };
            //          -  ^^^^ cannot infer type for `[_; 0]`
            //          |
            //          consider giving this closure parameter a type
            // ```
            //
            // After clearing, it looks something like this:
            // ```
            // let x = |_| {  };
            //          ^ consider giving this closure parameter a type
            // ```
            labels.clear();
            labels.push((pattern.span, format!("consider giving this closure parameter a type")));
        }

        if let Some(pattern) = local_visitor.found_local_pattern {
            if let Some(simple_name) = pattern.simple_name() {
                labels.push((pattern.span, format!("consider giving `{}` a type", simple_name)));
            } else {
                labels.push((pattern.span, format!("consider giving the pattern a type")));
            }
        }

        let mut err = struct_span_err!(self.tcx.sess,
                                       err_span,
                                       E0282,
                                       "type annotations needed");

        for (target_span, label_message) in labels {
            err.span_label(target_span, &label_message);
        }

        err.emit();
    }

    fn note_obligation_cause<T>(&self,
                                err: &mut DiagnosticBuilder,
                                obligation: &Obligation<'tcx, T>)
        where T: fmt::Display
    {
        self.note_obligation_cause_code(err,
                                        &obligation.predicate,
                                        &obligation.cause.code);
    }

    fn note_obligation_cause_code<T>(&self,
                                     err: &mut DiagnosticBuilder,
                                     predicate: &T,
                                     cause_code: &ObligationCauseCode<'tcx>)
        where T: fmt::Display
    {
        let tcx = self.tcx;
        match *cause_code {
            ObligationCauseCode::ExprAssignable |
            ObligationCauseCode::MatchExpressionArm { .. } |
            ObligationCauseCode::IfExpression |
            ObligationCauseCode::IfExpressionWithNoElse |
            ObligationCauseCode::EquatePredicate |
            ObligationCauseCode::MainFunctionType |
            ObligationCauseCode::StartFunctionType |
            ObligationCauseCode::IntrinsicType |
            ObligationCauseCode::MethodReceiver |
            ObligationCauseCode::ReturnNoExpression |
            ObligationCauseCode::MiscObligation => {
            }
            ObligationCauseCode::SliceOrArrayElem => {
                err.note("slice and array elements must have `Sized` type");
            }
            ObligationCauseCode::TupleElem => {
                err.note("tuple elements must have `Sized` type");
            }
            ObligationCauseCode::ProjectionWf(data) => {
                err.note(&format!("required so that the projection `{}` is well-formed",
                                  data));
            }
            ObligationCauseCode::ReferenceOutlivesReferent(ref_ty) => {
                err.note(&format!("required so that reference `{}` does not outlive its referent",
                                  ref_ty));
            }
            ObligationCauseCode::ObjectTypeBound(object_ty, region) => {
                err.note(&format!("required so that the lifetime bound of `{}` for `{}` \
                                   is satisfied",
                                  region, object_ty));
            }
            ObligationCauseCode::ItemObligation(item_def_id) => {
                let item_name = tcx.item_path_str(item_def_id);
                err.note(&format!("required by `{}`", item_name));
            }
            ObligationCauseCode::ObjectCastObligation(object_ty) => {
                err.note(&format!("required for the cast to the object type `{}`",
                                  self.ty_to_string(object_ty)));
            }
            ObligationCauseCode::RepeatVec => {
                err.note("the `Copy` trait is required because the \
                          repeated element will be copied");
            }
            ObligationCauseCode::VariableType(_) => {
                err.note("all local variables must have a statically known size");
            }
            ObligationCauseCode::ReturnType => {
                err.note("the return type of a function must have a \
                          statically known size");
            }
            ObligationCauseCode::AssignmentLhsSized => {
                err.note("the left-hand-side of an assignment must have a statically known size");
            }
            ObligationCauseCode::StructInitializerSized => {
                err.note("structs must have a statically known size to be initialized");
            }
            ObligationCauseCode::FieldSized => {
                err.note("only the last field of a struct may have a dynamically sized type");
            }
            ObligationCauseCode::ConstSized => {
                err.note("constant expressions must have a statically known size");
            }
            ObligationCauseCode::SharedStatic => {
                err.note("shared static variables must have a type that implements `Sync`");
            }
            ObligationCauseCode::BuiltinDerivedObligation(ref data) => {
                let parent_trait_ref = self.resolve_type_vars_if_possible(&data.parent_trait_ref);
                err.note(&format!("required because it appears within the type `{}`",
                                  parent_trait_ref.0.self_ty()));
                let parent_predicate = parent_trait_ref.to_predicate();
                self.note_obligation_cause_code(err,
                                                &parent_predicate,
                                                &data.parent_code);
            }
            ObligationCauseCode::ImplDerivedObligation(ref data) => {
                let parent_trait_ref = self.resolve_type_vars_if_possible(&data.parent_trait_ref);
                err.note(
                    &format!("required because of the requirements on the impl of `{}` for `{}`",
                             parent_trait_ref,
                             parent_trait_ref.0.self_ty()));
                let parent_predicate = parent_trait_ref.to_predicate();
                self.note_obligation_cause_code(err,
                                                &parent_predicate,
                                                &data.parent_code);
            }
            ObligationCauseCode::CompareImplMethodObligation { .. } => {
                err.note(
                    &format!("the requirement `{}` appears on the impl method \
                              but not on the corresponding trait method",
                             predicate));
            }
        }
    }

    fn suggest_new_overflow_limit(&self, err: &mut DiagnosticBuilder) {
        let current_limit = self.tcx.sess.recursion_limit.get();
        let suggested_limit = current_limit * 2;
        err.help(&format!(
                          "consider adding a `#![recursion_limit=\"{}\"]` attribute to your crate",
                          suggested_limit));
    }
}

