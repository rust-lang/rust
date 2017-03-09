// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Error Reporting Code for the inference engine
//!
//! Because of the way inference, and in particular region inference,
//! works, it often happens that errors are not detected until far after
//! the relevant line of code has been type-checked. Therefore, there is
//! an elaborate system to track why a particular constraint in the
//! inference graph arose so that we can explain to the user what gave
//! rise to a particular error.
//!
//! The basis of the system are the "origin" types. An "origin" is the
//! reason that a constraint or inference variable arose. There are
//! different "origin" enums for different kinds of constraints/variables
//! (e.g., `TypeOrigin`, `RegionVariableOrigin`). An origin always has
//! a span, but also more information so that we can generate a meaningful
//! error message.
//!
//! Having a catalogue of all the different reasons an error can arise is
//! also useful for other reasons, like cross-referencing FAQs etc, though
//! we are not really taking advantage of this yet.
//!
//! # Region Inference
//!
//! Region inference is particularly tricky because it always succeeds "in
//! the moment" and simply registers a constraint. Then, at the end, we
//! can compute the full graph and report errors, so we need to be able to
//! store and later report what gave rise to the conflicting constraints.
//!
//! # Subtype Trace
//!
//! Determining whether `T1 <: T2` often involves a number of subtypes and
//! subconstraints along the way. A "TypeTrace" is an extended version
//! of an origin that traces the types and other values that were being
//! compared. It is not necessarily comprehensive (in fact, at the time of
//! this writing it only tracks the root values being compared) but I'd
//! like to extend it to include significant "waypoints". For example, if
//! you are comparing `(T1, T2) <: (T3, T4)`, and the problem is that `T2
//! <: T4` fails, I'd like the trace to include enough information to say
//! "in the 2nd element of the tuple". Similarly, failures when comparing
//! arguments or return types in fn types should be able to cite the
//! specific position, etc.
//!
//! # Reality vs plan
//!
//! Of course, there is still a LOT of code in typeck that has yet to be
//! ported to this system, and which relies on string concatenation at the
//! time of error detection.

use infer;
use super::{InferCtxt, TypeTrace, SubregionOrigin, RegionVariableOrigin, ValuePairs};
use super::region_inference::{RegionResolutionError, ConcreteFailure, SubSupConflict,
                              GenericBoundFailure, GenericKind};

use std::fmt;
use hir;
use hir::map as hir_map;
use hir::def_id::DefId;
use middle::region;
use traits::{ObligationCause, ObligationCauseCode};
use ty::{self, TyCtxt, TypeFoldable};
use ty::{Region, Issue32330};
use ty::error::TypeError;
use syntax_pos::{Pos, Span};
use errors::DiagnosticBuilder;

mod note;

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn note_and_explain_region(self,
                                   err: &mut DiagnosticBuilder,
                                   prefix: &str,
                                   region: &'tcx ty::Region,
                                   suffix: &str) {
        fn item_scope_tag(item: &hir::Item) -> &'static str {
            match item.node {
                hir::ItemImpl(..) => "impl",
                hir::ItemStruct(..) => "struct",
                hir::ItemUnion(..) => "union",
                hir::ItemEnum(..) => "enum",
                hir::ItemTrait(..) => "trait",
                hir::ItemFn(..) => "function body",
                _ => "item"
            }
        }

        fn trait_item_scope_tag(item: &hir::TraitItem) -> &'static str {
            match item.node {
                hir::TraitItemKind::Method(..) => "method body",
                hir::TraitItemKind::Const(..) |
                hir::TraitItemKind::Type(..) => "associated item"
            }
        }

        fn impl_item_scope_tag(item: &hir::ImplItem) -> &'static str {
            match item.node {
                hir::ImplItemKind::Method(..) => "method body",
                hir::ImplItemKind::Const(..) |
                hir::ImplItemKind::Type(_) => "associated item"
            }
        }

        fn explain_span<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                        heading: &str, span: Span)
                                        -> (String, Option<Span>) {
            let lo = tcx.sess.codemap().lookup_char_pos_adj(span.lo);
            (format!("the {} at {}:{}", heading, lo.line, lo.col.to_usize()),
             Some(span))
        }

        let (description, span) = match *region {
            ty::ReScope(scope) => {
                let new_string;
                let unknown_scope = || {
                    format!("{}unknown scope: {:?}{}.  Please report a bug.",
                            prefix, scope, suffix)
                };
                let span = match scope.span(&self.region_maps, &self.hir) {
                    Some(s) => s,
                    None => {
                        err.note(&unknown_scope());
                        return;
                    }
                };
                let tag = match self.hir.find(scope.node_id(&self.region_maps)) {
                    Some(hir_map::NodeBlock(_)) => "block",
                    Some(hir_map::NodeExpr(expr)) => match expr.node {
                        hir::ExprCall(..) => "call",
                        hir::ExprMethodCall(..) => "method call",
                        hir::ExprMatch(.., hir::MatchSource::IfLetDesugar { .. }) => "if let",
                        hir::ExprMatch(.., hir::MatchSource::WhileLetDesugar) =>  "while let",
                        hir::ExprMatch(.., hir::MatchSource::ForLoopDesugar) =>  "for",
                        hir::ExprMatch(..) => "match",
                        _ => "expression",
                    },
                    Some(hir_map::NodeStmt(_)) => "statement",
                    Some(hir_map::NodeItem(it)) => item_scope_tag(&it),
                    Some(hir_map::NodeTraitItem(it)) => trait_item_scope_tag(&it),
                    Some(hir_map::NodeImplItem(it)) => impl_item_scope_tag(&it),
                    Some(_) | None => {
                        err.span_note(span, &unknown_scope());
                        return;
                    }
                };
                let scope_decorated_tag = match self.region_maps.code_extent_data(scope) {
                    region::CodeExtentData::Misc(_) => tag,
                    region::CodeExtentData::CallSiteScope { .. } => {
                        "scope of call-site for function"
                    }
                    region::CodeExtentData::ParameterScope { .. } => {
                        "scope of function body"
                    }
                    region::CodeExtentData::DestructionScope(_) => {
                        new_string = format!("destruction scope surrounding {}", tag);
                        &new_string[..]
                    }
                    region::CodeExtentData::Remainder(r) => {
                        new_string = format!("block suffix following statement {}",
                                             r.first_statement_index);
                        &new_string[..]
                    }
                };
                explain_span(self, scope_decorated_tag, span)
            }

            ty::ReFree(ref fr) => {
                let prefix = match fr.bound_region {
                    ty::BrAnon(idx) => {
                        format!("the anonymous lifetime #{} defined on", idx + 1)
                    }
                    ty::BrFresh(_) => "an anonymous lifetime defined on".to_owned(),
                    _ => {
                        format!("the lifetime {} as defined on",
                                fr.bound_region)
                    }
                };

                let node = fr.scope.node_id(&self.region_maps);
                let unknown;
                let tag = match self.hir.find(node) {
                    Some(hir_map::NodeBlock(_)) |
                    Some(hir_map::NodeExpr(_)) => "body",
                    Some(hir_map::NodeItem(it)) => item_scope_tag(&it),
                    Some(hir_map::NodeTraitItem(it)) => trait_item_scope_tag(&it),
                    Some(hir_map::NodeImplItem(it)) => impl_item_scope_tag(&it),

                    // this really should not happen, but it does:
                    // FIXME(#27942)
                    Some(_) => {
                        unknown = format!("unexpected node ({}) for scope {:?}.  \
                                           Please report a bug.",
                                          self.hir.node_to_string(node), fr.scope);
                        &unknown
                    }
                    None => {
                        unknown = format!("unknown node for scope {:?}.  \
                                           Please report a bug.", fr.scope);
                        &unknown
                    }
                };
                let (msg, opt_span) = explain_span(self, tag, self.hir.span(node));
                (format!("{} {}", prefix, msg), opt_span)
            }

            ty::ReStatic => ("the static lifetime".to_owned(), None),

            ty::ReEmpty => ("the empty lifetime".to_owned(), None),

            ty::ReEarlyBound(ref data) => (data.name.to_string(), None),

            // FIXME(#13998) ReSkolemized should probably print like
            // ReFree rather than dumping Debug output on the user.
            //
            // We shouldn't really be having unification failures with ReVar
            // and ReLateBound though.
            ty::ReSkolemized(..) |
            ty::ReVar(_) |
            ty::ReLateBound(..) |
            ty::ReErased => {
                (format!("lifetime {:?}", region), None)
            }
        };
        let message = format!("{}{}{}", prefix, description, suffix);
        if let Some(span) = span {
            err.span_note(span, &message);
        } else {
            err.note(&message);
        }
    }
}

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    pub fn report_region_errors(&self,
                                errors: &Vec<RegionResolutionError<'tcx>>) {
        debug!("report_region_errors(): {} errors to start", errors.len());

        // try to pre-process the errors, which will group some of them
        // together into a `ProcessedErrors` group:
        let errors = self.process_errors(errors);

        debug!("report_region_errors: {} errors after preprocessing", errors.len());

        for error in errors {
            debug!("report_region_errors: error = {:?}", error);
            match error.clone() {
                ConcreteFailure(origin, sub, sup) => {
                    self.report_concrete_failure(origin, sub, sup).emit();
                }

                GenericBoundFailure(kind, param_ty, sub) => {
                    self.report_generic_bound_failure(kind, param_ty, sub);
                }

                SubSupConflict(var_origin,
                               sub_origin, sub_r,
                               sup_origin, sup_r) => {
                    self.report_sub_sup_conflict(var_origin,
                                                 sub_origin, sub_r,
                                                 sup_origin, sup_r);
                }
            }
        }
    }

    // This method goes through all the errors and try to group certain types
    // of error together, for the purpose of suggesting explicit lifetime
    // parameters to the user. This is done so that we can have a more
    // complete view of what lifetimes should be the same.
    // If the return value is an empty vector, it means that processing
    // failed (so the return value of this method should not be used).
    //
    // The method also attempts to weed out messages that seem like
    // duplicates that will be unhelpful to the end-user. But
    // obviously it never weeds out ALL errors.
    fn process_errors(&self, errors: &Vec<RegionResolutionError<'tcx>>)
                      -> Vec<RegionResolutionError<'tcx>> {
        debug!("process_errors()");

        // We want to avoid reporting generic-bound failures if we can
        // avoid it: these have a very high rate of being unhelpful in
        // practice. This is because they are basically secondary
        // checks that test the state of the region graph after the
        // rest of inference is done, and the other kinds of errors
        // indicate that the region constraint graph is internally
        // inconsistent, so these test results are likely to be
        // meaningless.
        //
        // Therefore, we filter them out of the list unless they are
        // the only thing in the list.

        let is_bound_failure = |e: &RegionResolutionError<'tcx>| match *e {
            ConcreteFailure(..) => false,
            SubSupConflict(..) => false,
            GenericBoundFailure(..) => true,
        };

        if errors.iter().all(|e| is_bound_failure(e)) {
            errors.clone()
        } else {
            errors.iter().filter(|&e| !is_bound_failure(e)).cloned().collect()
        }
    }

    /// Adds a note if the types come from similarly named crates
    fn check_and_note_conflicting_crates(&self,
                                         err: &mut DiagnosticBuilder,
                                         terr: &TypeError<'tcx>,
                                         sp: Span) {
        let report_path_match = |err: &mut DiagnosticBuilder, did1: DefId, did2: DefId| {
            // Only external crates, if either is from a local
            // module we could have false positives
            if !(did1.is_local() || did2.is_local()) && did1.krate != did2.krate {
                let exp_path = self.tcx.item_path_str(did1);
                let found_path = self.tcx.item_path_str(did2);
                // We compare strings because DefPath can be different
                // for imported and non-imported crates
                if exp_path == found_path {
                    let crate_name = self.tcx.sess.cstore.crate_name(did1.krate);
                    err.span_note(sp, &format!("Perhaps two different versions \
                                                of crate `{}` are being used?",
                                               crate_name));
                }
            }
        };
        match *terr {
            TypeError::Sorts(ref exp_found) => {
                // if they are both "path types", there's a chance of ambiguity
                // due to different versions of the same crate
                match (&exp_found.expected.sty, &exp_found.found.sty) {
                    (&ty::TyAdt(exp_adt, _), &ty::TyAdt(found_adt, _)) => {
                        report_path_match(err, exp_adt.did, found_adt.did);
                    },
                    _ => ()
                }
            },
            TypeError::Traits(ref exp_found) => {
                report_path_match(err, exp_found.expected, exp_found.found);
            },
            _ => () // FIXME(#22750) handle traits and stuff
        }
    }

    fn note_error_origin(&self,
                         err: &mut DiagnosticBuilder<'tcx>,
                         cause: &ObligationCause<'tcx>)
    {
        match cause.code {
            ObligationCauseCode::MatchExpressionArm { arm_span, source } => match source {
                hir::MatchSource::IfLetDesugar {..} => {
                    err.span_note(arm_span, "`if let` arm with an incompatible type");
                }
                _ => {
                    err.span_note(arm_span, "match arm with an incompatible type");
                }
            },
            _ => ()
        }
    }

    pub fn note_type_err(&self,
                         diag: &mut DiagnosticBuilder<'tcx>,
                         cause: &ObligationCause<'tcx>,
                         secondary_span: Option<(Span, String)>,
                         values: Option<ValuePairs<'tcx>>,
                         terr: &TypeError<'tcx>)
    {
        let (expected_found, is_simple_error) = match values {
            None => (None, false),
            Some(values) => {
                let is_simple_error = match values {
                    ValuePairs::Types(exp_found) => {
                        exp_found.expected.is_primitive() && exp_found.found.is_primitive()
                    }
                    _ => false,
                };
                let vals = match self.values_str(&values) {
                    Some((expected, found)) => Some((expected, found)),
                    None => {
                        // Derived error. Cancel the emitter.
                        self.tcx.sess.diagnostic().cancel(diag);
                        return
                    }
                };
                (vals, is_simple_error)
            }
        };

        let span = cause.span;

        if let Some((expected, found)) = expected_found {
            match (terr, is_simple_error, expected == found) {
                (&TypeError::Sorts(ref values), false,  true) => {
                    diag.note_expected_found_extra(
                        &"type", &expected, &found,
                        &format!(" ({})", values.expected.sort_string(self.tcx)),
                        &format!(" ({})", values.found.sort_string(self.tcx)));
                }
                (_, false,  _) => {
                    diag.note_expected_found(&"type", &expected, &found);
                }
                _ => (),
            }
        }

        diag.span_label(span, &terr);
        if let Some((sp, msg)) = secondary_span {
            diag.span_label(sp, &msg);
        }

        self.note_error_origin(diag, &cause);
        self.check_and_note_conflicting_crates(diag, terr, span);
        self.tcx.note_and_explain_type_err(diag, terr, span);
    }

    pub fn note_issue_32330(&self,
                            diag: &mut DiagnosticBuilder<'tcx>,
                            terr: &TypeError<'tcx>)
    {
        debug!("note_issue_32330: terr={:?}", terr);
        match *terr {
            TypeError::RegionsInsufficientlyPolymorphic(_, &Region::ReVar(vid)) |
            TypeError::RegionsOverlyPolymorphic(_, &Region::ReVar(vid)) => {
                match self.region_vars.var_origin(vid) {
                    RegionVariableOrigin::EarlyBoundRegion(_, _, Some(Issue32330 {
                        fn_def_id,
                        region_name
                    })) => {
                        diag.note(
                            &format!("lifetime parameter `{0}` declared on fn `{1}` \
                                      appears only in the return type, \
                                      but here is required to be higher-ranked, \
                                      which means that `{0}` must appear in both \
                                      argument and return types",
                                     region_name,
                                     self.tcx.item_path_str(fn_def_id)));
                        diag.note(
                            &format!("this error is the result of a recent bug fix; \
                                      for more information, see issue #33685 \
                                      <https://github.com/rust-lang/rust/issues/33685>"));
                    }
                    _ => { }
                }
            }
            _ => { }
        }
    }

    pub fn report_and_explain_type_error(&self,
                                         trace: TypeTrace<'tcx>,
                                         terr: &TypeError<'tcx>)
                                         -> DiagnosticBuilder<'tcx>
    {
        let span = trace.cause.span;
        let failure_str = trace.cause.as_failure_str();
        let mut diag = match trace.cause.code {
            ObligationCauseCode::IfExpressionWithNoElse => {
                struct_span_err!(self.tcx.sess, span, E0317, "{}", failure_str)
            }
            ObligationCauseCode::MainFunctionType => {
                struct_span_err!(self.tcx.sess, span, E0580, "{}", failure_str)
            }
            _ => {
                struct_span_err!(self.tcx.sess, span, E0308, "{}", failure_str)
            }
        };
        self.note_type_err(&mut diag, &trace.cause, None, Some(trace.values), terr);
        self.note_issue_32330(&mut diag, terr);
        diag
    }

    /// Returns a string of the form "expected `{}`, found `{}`".
    fn values_str(&self, values: &ValuePairs<'tcx>) -> Option<(String, String)> {
        match *values {
            infer::Types(ref exp_found) => self.expected_found_str(exp_found),
            infer::TraitRefs(ref exp_found) => self.expected_found_str(exp_found),
            infer::PolyTraitRefs(ref exp_found) => self.expected_found_str(exp_found),
        }
    }

    fn expected_found_str<T: fmt::Display + TypeFoldable<'tcx>>(
        &self,
        exp_found: &ty::error::ExpectedFound<T>)
        -> Option<(String, String)>
    {
        let exp_found = self.resolve_type_vars_if_possible(exp_found);
        if exp_found.references_error() {
            return None;
        }

        Some((format!("{}", exp_found.expected), format!("{}", exp_found.found)))
    }

    fn report_generic_bound_failure(&self,
                                    origin: SubregionOrigin<'tcx>,
                                    bound_kind: GenericKind<'tcx>,
                                    sub: &'tcx Region)
    {
        // FIXME: it would be better to report the first error message
        // with the span of the parameter itself, rather than the span
        // where the error was detected. But that span is not readily
        // accessible.

        let labeled_user_string = match bound_kind {
            GenericKind::Param(ref p) =>
                format!("the parameter type `{}`", p),
            GenericKind::Projection(ref p) =>
                format!("the associated type `{}`", p),
        };

        if let SubregionOrigin::CompareImplMethodObligation {
            span, item_name, impl_item_def_id, trait_item_def_id, lint_id
        } = origin {
            self.report_extra_impl_obligation(span,
                                              item_name,
                                              impl_item_def_id,
                                              trait_item_def_id,
                                              &format!("`{}: {}`", bound_kind, sub),
                                              lint_id)
                .emit();
            return;
        }

        let mut err = match *sub {
            ty::ReFree(ty::FreeRegion {bound_region: ty::BrNamed(..), ..}) => {
                // Does the required lifetime have a nice name we can print?
                let mut err = struct_span_err!(self.tcx.sess,
                                               origin.span(),
                                               E0309,
                                               "{} may not live long enough",
                                               labeled_user_string);
                err.help(&format!("consider adding an explicit lifetime bound `{}: {}`...",
                         bound_kind,
                         sub));
                err
            }

            ty::ReStatic => {
                // Does the required lifetime have a nice name we can print?
                let mut err = struct_span_err!(self.tcx.sess,
                                               origin.span(),
                                               E0310,
                                               "{} may not live long enough",
                                               labeled_user_string);
                err.help(&format!("consider adding an explicit lifetime \
                                   bound `{}: 'static`...",
                                  bound_kind));
                err
            }

            _ => {
                // If not, be less specific.
                let mut err = struct_span_err!(self.tcx.sess,
                                               origin.span(),
                                               E0311,
                                               "{} may not live long enough",
                                               labeled_user_string);
                err.help(&format!("consider adding an explicit lifetime bound for `{}`",
                                  bound_kind));
                self.tcx.note_and_explain_region(
                    &mut err,
                    &format!("{} must be valid for ", labeled_user_string),
                    sub,
                    "...");
                err
            }
        };

        self.note_region_origin(&mut err, &origin);
        err.emit();
    }

    fn report_sub_sup_conflict(&self,
                               var_origin: RegionVariableOrigin,
                               sub_origin: SubregionOrigin<'tcx>,
                               sub_region: &'tcx Region,
                               sup_origin: SubregionOrigin<'tcx>,
                               sup_region: &'tcx Region) {
        let mut err = self.report_inference_failure(var_origin);

        self.tcx.note_and_explain_region(&mut err,
            "first, the lifetime cannot outlive ",
            sup_region,
            "...");

        self.note_region_origin(&mut err, &sup_origin);

        self.tcx.note_and_explain_region(&mut err,
            "but, the lifetime must be valid for ",
            sub_region,
            "...");

        self.note_region_origin(&mut err, &sub_origin);
        err.emit();
    }
}

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    fn report_inference_failure(&self,
                                var_origin: RegionVariableOrigin)
                                -> DiagnosticBuilder<'tcx> {
        let br_string = |br: ty::BoundRegion| {
            let mut s = br.to_string();
            if !s.is_empty() {
                s.push_str(" ");
            }
            s
        };
        let var_description = match var_origin {
            infer::MiscVariable(_) => "".to_string(),
            infer::PatternRegion(_) => " for pattern".to_string(),
            infer::AddrOfRegion(_) => " for borrow expression".to_string(),
            infer::Autoref(_) => " for autoref".to_string(),
            infer::Coercion(_) => " for automatic coercion".to_string(),
            infer::LateBoundRegion(_, br, infer::FnCall) => {
                format!(" for lifetime parameter {}in function call",
                        br_string(br))
            }
            infer::LateBoundRegion(_, br, infer::HigherRankedType) => {
                format!(" for lifetime parameter {}in generic type", br_string(br))
            }
            infer::LateBoundRegion(_, br, infer::AssocTypeProjection(type_name)) => {
                format!(" for lifetime parameter {}in trait containing associated type `{}`",
                        br_string(br), type_name)
            }
            infer::EarlyBoundRegion(_, name, _) => {
                format!(" for lifetime parameter `{}`",
                        name)
            }
            infer::BoundRegionInCoherence(name) => {
                format!(" for lifetime parameter `{}` in coherence check",
                        name)
            }
            infer::UpvarRegion(ref upvar_id, _) => {
                format!(" for capture of `{}` by closure",
                        self.tcx.local_var_name_str(upvar_id.var_id).to_string())
            }
        };

        struct_span_err!(self.tcx.sess, var_origin.span(), E0495,
                  "cannot infer an appropriate lifetime{} \
                   due to conflicting requirements",
                  var_description)
    }
}

impl<'tcx> ObligationCause<'tcx> {
    fn as_failure_str(&self) -> &'static str {
        use traits::ObligationCauseCode::*;
        match self.code {
            CompareImplMethodObligation { .. } => "method not compatible with trait",
            MatchExpressionArm { source, .. } => match source {
                hir::MatchSource::IfLetDesugar{..} => "`if let` arms have incompatible types",
                _ => "match arms have incompatible types",
            },
            IfExpression => "if and else have incompatible types",
            IfExpressionWithNoElse => "if may be missing an else clause",
            EquatePredicate => "equality predicate not satisfied",
            MainFunctionType => "main function has wrong type",
            StartFunctionType => "start function has wrong type",
            IntrinsicType => "intrinsic has wrong type",
            MethodReceiver => "mismatched method receiver",
            _ => "mismatched types",
        }
    }

    fn as_requirement_str(&self) -> &'static str {
        use traits::ObligationCauseCode::*;
        match self.code {
            CompareImplMethodObligation { .. } => "method type is compatible with trait",
            ExprAssignable => "expression is assignable",
            MatchExpressionArm { source, .. } => match source {
                hir::MatchSource::IfLetDesugar{..} => "`if let` arms have compatible types",
                _ => "match arms have compatible types",
            },
            IfExpression => "if and else have compatible types",
            IfExpressionWithNoElse => "if missing an else returns ()",
            EquatePredicate => "equality where clause is satisfied",
            MainFunctionType => "`main` function has the correct type",
            StartFunctionType => "`start` function has the correct type",
            IntrinsicType => "intrinsic has the correct type",
            MethodReceiver => "method receiver has the correct type",
            _ => "types are compatible",
        }
    }
}
